"""Combine learned actors by hand using the same live physical observation.

Weights are fixed or follow an explicitly declared source-clock schedule.
Objects, source timing, physics and reset behavior remain unchanged.
"""
import json
import os
from pathlib import Path
import numpy as np


def weights_for_layout(layout, profile):
    weights=np.zeros(layout.action_dim,np.float32)
    if set(profile)!={'left','right'} or any(not 0.<=float(v)<=1. for v in profile.values()):
        raise ValueError('Declare fixed left/right alternate-actor weights in [0, 1]')
    if set(layout.sides)!={'left','right'}:raise ValueError('Both hand identities are required')
    for side,start,count in zip(layout.sides,layout.hand_action_starts,layout.finger_counts):
        weights[start:start+6+count]=profile[side]
    return weights


def scheduled_weights(layout,spec,native_frames,source_frames):
    table=np.broadcast_to(weights_for_layout(layout,spec['hand_weights']),(native_frames,layout.action_dim)).copy()
    source=np.linspace(0,source_frames-1,native_frames)
    for side,item in spec.get('source_schedule',{}).items():
        assert side in layout.sides and set(item)=={'initial_weight','start_source_frame','ramp_source_frames'}
        initial=float(item['initial_weight']);start=float(item['start_source_frame']);ramp=float(item['ramp_source_frames'])
        assert 0<=initial<=1 and 0<=start<source_frames-1 and 0<ramp<=source_frames-1-start
        w=np.clip((source-start)/ramp,0,1);w=w*w*(3-2*w);h=layout.sides.index(side);begin=layout.hand_action_starts[h]
        table[:,begin:begin+6+layout.finger_counts[h]]=(initial+w*(spec['hand_weights'][side]-initial))[:,None]
    return table


def install():
    raw=os.environ.get('V2D_ACTOR_COMBINATION')
    if not raw:return
    spec=json.loads(raw)
    alternate=Path(spec['alternate_checkpoint'])
    if not alternate.is_file():raise ValueError('Alternate actor checkpoint is missing')
    import hashlib
    if hashlib.sha256(alternate.read_bytes()).hexdigest()!=spec['alternate_checkpoint_sha256']:
        raise ValueError('Alternate actor checkpoint hash mismatch')
    import jax
    import jax.numpy as jnp
    from omegaconf import OmegaConf
    from flash_chord.envs.base import BaseEnv
    if spec.get('finger_bias_rad') is not None or spec.get('finger_bias_profiles') is not None:
        assert not os.environ.get('V2D_GRIP_BIAS') and not os.environ.get('V2D_GRIP_BIAS_PROFILES')
        key='V2D_GRIP_BIAS' if spec.get('finger_bias_rad') is not None else 'V2D_GRIP_BIAS_PROFILES'
        assert not ('finger_bias_rad' in spec and 'finger_bias_profiles' in spec)
        os.environ[key]=json.dumps(spec.get('finger_bias_rad',spec.get('finger_bias_profiles')))
        from grip_bias_controller import install as grip
        grip()
    from flash_chord.configuration import instantiate_typed
    from flash_chord.training.checkpoint import read_checkpoint_metadata
    from flash_chord.training.checkpoint_metadata import checkpoint_config,checkpoint_policy_schema,validate_policy_schema
    from flash_chord.training.flash_sac.config import TrainingConfig
    import flash_chord.training.flash_sac.evaluation as inference
    if getattr(BaseEnv,'_actor_combination_installed',False):raise RuntimeError('Actor combination installed twice')
    original_init=BaseEnv.__init__;layout_holder={}
    def initialize(self,*args,**kwargs):
        original_init(self,*args,**kwargs)
        layout_holder['layout']=self.action.buffers.layout;layout_holder['env']=self
    BaseEnv.__init__=initialize;BaseEnv._actor_combination_installed=True
    native_loader=inference.load_policy_for_inference
    def load(checkpoint,training,observation,action_dim,*,deterministic,device=None):
        if not deterministic:raise ValueError('Actor combinations require deterministic evaluation')
        primary,primary_action,key,metadata=native_loader(checkpoint,training,observation,action_dim,deterministic=True,device=device)
        alternate_meta=read_checkpoint_metadata(alternate)
        validate_policy_schema(checkpoint_policy_schema(metadata),checkpoint_policy_schema(alternate_meta))
        config=checkpoint_config(alternate_meta)
        other_training=instantiate_typed(OmegaConf.create(config['training']),TrainingConfig)
        other,other_action,other_key,_=native_loader(str(alternate),other_training,observation,action_dim,deterministic=True,device=device)
        weights=jax.device_put(jnp.asarray(weights_for_layout(layout_holder['layout'],spec['hand_weights'])),device)
        combine=jax.jit(lambda a,b:a*(1.-weights)+b*weights).lower(
            jnp.zeros_like(observation[:,:action_dim]),jnp.zeros_like(observation[:,:action_dim])).compile()
        timed=bool(spec.get('source_schedule'))
        if timed:
            env=layout_holder['env'];cfg=json.loads(Path('/episode/episode-config.json').read_text())
            table=scheduled_weights(layout_holder['layout'],spec,env.reference.num_frames,cfg['frames'])
            dynamic=jax.jit(lambda a,b,w:a*(1.-w)+b*w).lower(jnp.zeros_like(observation[:,:action_dim]),jnp.zeros_like(observation[:,:action_dim]),jnp.zeros_like(observation[:,:action_dim])).compile()
        def action(current,current_observation,current_key):
            a,next_key=primary_action(current,current_observation,current_key)
            b,_=other_action(other,current_observation,other_key)
            if timed:
                frames=np.clip(env.timestep.numpy(),0,len(table)-1);w=jax.device_put(jnp.asarray(table[frames]),device)
                return dynamic(a,b,w),next_key
            return combine(a,b),next_key
        return primary,action,key,metadata
    inference.load_policy_for_inference=load
    # The evaluator may already hold its imported alias.
    import flash_chord.evaluation.harness as harness
    harness.load_flash_sac_policy=load
    print('Declared source-phase learned actor combination:' if spec.get('source_schedule') else 'Declared fixed whole-attempt learned actor combination:',spec,flush=True)
