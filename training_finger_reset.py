"""Training-only hand initialization: half exact reference poses, half scaled finger poses.

Inference keeps NVIDIA's native exact-reference explicit reset. No object state,
force, timing, action, or reference trajectory is changed by this extension.
"""
import warp as wp
import numpy as np
from flash_chord.lifecycle.reset import ReferenceReset,ReferenceResetConfig

@wp.kernel
def select_exact_fingers(mask:wp.array(dtype=wp.int32),counts:wp.array(dtype=wp.int32),
                         seed:int,factors:wp.array(dtype=wp.float32)):
    world=wp.tid()
    if mask[world]!=0:
        rng=wp.rand_init(seed+7919,world+counts[world]*104729)
        if wp.randf(rng)<0.5:factors[world]=1.0

@wp.kernel
def count_opening_outcomes(mask:wp.array(dtype=wp.int32),done:wp.array(dtype=wp.int32),
        truncation:wp.array(dtype=wp.int32),frame:wp.array(dtype=wp.int32),
        start:wp.array(dtype=wp.int32),factor:wp.array(dtype=wp.float32),
        minimum:wp.array(dtype=wp.int32),gate_end:int,counts:wp.array(dtype=wp.int32)):
    world=wp.tid()
    if mask[world]!=0 and (done[world]!=0 or truncation[world]!=0) and start[world]==minimum[0] and factor[world]==1.0:
        wp.atomic_add(counts,0,1)
        if frame[world]>gate_end-1 or (frame[world]==gate_end-1 and truncation[world]!=0):
            wp.atomic_add(counts,1,1)

@wp.kernel
def clamp_reset_frame(mask:wp.array(dtype=wp.int32),minimum:wp.array(dtype=wp.int32),frames:wp.array(dtype=wp.int32)):
    world=wp.tid()
    if mask[world]!=0:frames[world]=wp.max(frames[world],minimum[0])

@wp.kernel
def overlapping_reset_starts(mask:wp.array(dtype=wp.int32),counts:wp.array(dtype=wp.int32),
        minimum:wp.array(dtype=wp.int32),seed:int,frames:wp.array(dtype=wp.int32)):
    world=wp.tid()
    if mask[world]!=0 and minimum[0]>0 and frames[world]==minimum[0]:
        rng=wp.rand_init(seed+15427,world+counts[world]*65537)
        draw=wp.randf(rng)
        if draw<0.1:frames[world]=0
        elif draw<0.3:frames[world]=wp.max(0,minimum[0]-4)

@wp.kernel
def focus_reset_starts(mask:wp.array(dtype=wp.int32),counts:wp.array(dtype=wp.int32),
        probability:wp.array(dtype=wp.float32),interval:wp.array(dtype=wp.int32),seed:int,frames:wp.array(dtype=wp.int32)):
    world=wp.tid()
    if mask[world]!=0:
        rng=wp.rand_init(seed+27059,world+counts[world]*65537)
        if wp.randf(rng)<probability[0]:frames[world]=wp.randi(rng,interval[0],interval[1]+1)

class MixedFingerReferenceReset(ReferenceReset):
    exact_reference_finger_probability=0.5

    def sample(self,reset_mask,curriculum_voc_scale):
        if getattr(self,'adaptive_enabled',False) and getattr(self,'environment',None) is not None:
            env=self.environment
            wp.launch(count_opening_outcomes,dim=self.world_count,
                inputs=[reset_mask,env.done,env.truncation,env.timestep,env.episode_start_frame,self.finger_scale,
                        self.minimum_frame,self.opening_gate_end],outputs=[self.opening_counts],device=self.finger_scale.device)
        super().sample(reset_mask,curriculum_voc_scale)
        if getattr(self,'focus_enabled',False):
            wp.launch(focus_reset_starts,dim=self.world_count,
                inputs=[reset_mask,self.reset_count,self.focus_probability,self.focus_interval,self.config.seed],
                outputs=[self.reset_frame],device=self.finger_scale.device)
        if getattr(self,'adaptive_enabled',False):
            wp.launch(clamp_reset_frame,dim=self.world_count,inputs=[reset_mask,self.minimum_frame],
                      outputs=[self.reset_frame],device=self.finger_scale.device)
            wp.launch(overlapping_reset_starts,dim=self.world_count,
                inputs=[reset_mask,self.reset_count,self.minimum_frame,self.config.seed],
                outputs=[self.reset_frame],device=self.finger_scale.device)
        wp.launch(select_exact_fingers,dim=self.world_count,
                  inputs=[reset_mask,self.reset_count,self.config.seed],outputs=[self.finger_scale],
                  device=self.finger_scale.device)

    def configure_focus(self,minimum_frame,maximum_frame,probability):
        assert 0<=minimum_frame<=maximum_frame<self.num_frames and 0.<=probability<=1.
        assert not getattr(self,'adaptive_enabled',False)
        self.focus_enabled=True;self.focus_lo=int(minimum_frame);self.focus_hi=int(maximum_frame)
        self.focus_probability=wp.array([probability],dtype=wp.float32,device=self.finger_scale.device)
        self.focus_interval=wp.array([minimum_frame,maximum_frame],dtype=wp.int32,device=self.finger_scale.device)

    def set_focus_interval(self,minimum_frame,maximum_frame):
        assert self.focus_enabled and 0<=minimum_frame<=maximum_frame<self.num_frames
        self.focus_lo=int(minimum_frame);self.focus_hi=int(maximum_frame)
        self.focus_interval.assign(np.array([minimum_frame,maximum_frame],np.int32))

    def configure_adaptive(self,minimum_frame,opening_gate_end,stride=2):
        assert 0<=minimum_frame<opening_gate_end<=self.num_frames
        self.adaptive_enabled=True;self.opening_gate_end=int(opening_gate_end);self.stride=int(stride)
        self.minimum_frame=wp.array([minimum_frame],dtype=wp.int32,device=self.finger_scale.device)
        self.opening_counts=wp.zeros(2,dtype=wp.int32,device=self.finger_scale.device)
        self.last_adaptive_check_step=None

    def maybe_advance(self,step):
        if not getattr(self,'adaptive_enabled',False):return None
        if self.last_adaptive_check_step is None:
            self.last_adaptive_check_step=int(step);self.opening_counts.zero_();return None
        if step-self.last_adaptive_check_step<100_000:return None
        total,passed=map(int,self.opening_counts.numpy())
        if total<128:return None
        before=int(self.minimum_frame.numpy()[0]);score=passed/total
        after=max(0,before-self.stride) if score>=.9 else before
        if after!=before:self.minimum_frame.assign(np.array([after],dtype=np.int32))
        self.opening_counts.zero_();self.last_adaptive_check_step=int(step)
        return {'environment_steps':int(step),'previous_minimum_reference_frame':before,
            'minimum_reference_frame':after,'exact_initialization_episodes':total,
            'opening_success_fraction_in_training':score,'advanced':after!=before,
            'interpretation':'Training curriculum only; canonical independent evaluation always starts at zero.'}

def install(native_globals,adaptive=None,focus=None):
    assert not (adaptive and focus)
    native_env=native_globals['RLEnv']
    class TrainingRLEnv(native_env):
        def __init__(self,scene,reference,config=None,**kwargs):
            assert config is not None and isinstance(config.reset,ReferenceResetConfig)
            assert config.reset.reset_finger_openness==1.0 and config.reset.voc_decay_steps==0
            assert config.voc.max_force==0. and config.voc.max_torque==0.
            if kwargs.get('reset_policy') is None:
                kwargs['reset_policy']=MixedFingerReferenceReset.build(scene.world_count,reference.num_frames,
                    config=config.reset,device=kwargs.get('device') or scene.model.device)
                if adaptive:kwargs['reset_policy'].configure_adaptive(**adaptive)
                if focus:
                    values=dict(focus);source=values.pop('source_interval')
                    values.update(minimum_frame=round(source[0]*(reference.num_frames-1)/(focus['source_frame_count']-1)),
                        maximum_frame=round(source[1]*(reference.num_frames-1)/(focus['source_frame_count']-1)))
                    values.pop('source_frame_count');kwargs['reset_policy'].configure_focus(**values)
            super().__init__(scene,reference,config=config,**kwargs)
            self.reset_policy.environment=self
    native_globals['RLEnv']=TrainingRLEnv
