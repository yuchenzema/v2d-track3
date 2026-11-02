"""Diagnostic: sample the learned SAC distribution instead of using its mean action."""
import json,os,runpy
from pathlib import Path
import flash_chord.evaluation.harness as harness

mode=os.environ.get('V2D_POLICY_SAMPLE_MODE','iid')
assert mode in ['mean','iid','zeta']
open_finger_reset=os.environ.get('V2D_OPEN_FINGER_RESET')=='1'
if open_finger_reset:
    import warp as wp
    from flash_chord.lifecycle.reset import ReferenceReset
    native_reset=ReferenceReset.prepare_explicit
    @wp.kernel
    def open_selected(mask:wp.array(dtype=wp.int32),factor:wp.array(dtype=wp.float32)):
        world=wp.tid()
        if mask[world]!=0:factor[world]=0.
    def reset(self,reset_mask,curriculum_voc_scale):
        native_reset(self,reset_mask,curriculum_voc_scale)
        wp.launch(open_selected,dim=self.world_count,inputs=[reset_mask],outputs=[self.finger_scale])
    ReferenceReset.prepare_explicit=reset
original=harness.load_flash_sac_policy

def load(*args,**kwargs):
    if mode=='mean':return original(*args,**kwargs)
    if mode=='iid':
        kwargs['deterministic']=False
        return original(*args,**kwargs)
    import jax
    from flash_chord.training.flash_sac.evaluation import create_actor_template
    from flash_chord.training.flash_sac.exploration import create_exploration,collection_action,truncated_zeta_cdf
    state,_,key,metadata=original(*args,**kwargs)
    training=args[1];observation=args[2];action_dim=args[3]
    actor,_=create_actor_template(training,observation.shape[1],action_dim,device=kwargs.get('device'))
    exploration=create_exploration(observation.shape[0],action_dim,device=kwargs.get('device'))
    cdf=truncated_zeta_cdf(training.exploration.zeta_exponent,training.exploration.maximum_noise_repeat)
    def sample(current,obs,current_key,noise):
        return collection_action(actor,{'params':current.params,'batch_stats':current.batch_stats},obs,noise,current_key,cdf,
            stochastic=True,action_transform=training.network.action_transform,action_scale=training.network.action_scale)
    compiled=jax.jit(sample)
    def action(current,obs,current_key):
        nonlocal exploration
        result,exploration,next_key=compiled(current,obs,current_key,exploration)
        return result,next_key
    return state,action,key,metadata

harness.load_flash_sac_policy=load
runpy.run_path('/workspace/scripts/evaluate_policy.py',run_name='__main__')
path=Path(os.environ['V2D_SAMPLE_PROBE_OUTPUT'])
result=json.loads(path.read_text())
result.update(diagnostic_only=True,full_human_motion_match_verified=False,
    declared_policy_action_sampling=mode,policy_action_is_mean=(mode=='mean'),
    declared_initial_robot_finger_scale=0. if open_finger_reset else 1.,
    object_assistance_zero=True,source_clock_unchanged=True)
path.write_text(json.dumps(result,indent=2))
