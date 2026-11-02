"""Declared finger flexion corrections through the ordinary robot action mapping.

Only flexion motor commands change. Original object poses, geometry, timing,
forces and reset rules are untouched. Grasp intervals come from the source
reference already used by the learned policy.
"""
import json
import os
from pathlib import Path
import numpy as np
import warp as wp
from flash_chord.envs.base import BaseEnv
from flash_chord.runtime.actions.hand_pose import ResidualHandPoseAction


@wp.kernel
def apply_bias(action:wp.array(dtype=wp.float32),frame:wp.array(dtype=wp.int32),
               offsets:wp.array3d(dtype=wp.float32),scale:wp.array(dtype=wp.float32),
               input_scale:wp.array(dtype=wp.float32),mapping:int,frames:int,
               dimension:int,out:wp.array(dtype=wp.float32)):
    index=wp.tid();world=index//dimension;component=index%dimension
    bias=offsets[wp.clamp(frame[world],0,frames-1),world,component]
    value=action[index]
    if bias!=0.:
        s=input_scale[component];mapped=value
        if mapping==1:
            mapped=s*wp.clamp(value,-1.,1.)
        elif mapping==2:
            a=wp.clamp(value,-1.,1.)
            mapped=a/((1.-wp.abs(a))+wp.abs(a)/s)
        mapped=mapped+bias/scale[component]
        if mapping==1:
            value=wp.clamp(mapped/s,-1.,1.)
        elif mapping==2:
            mapped=wp.clamp(mapped,-s,s)
            value=mapped/(1.+wp.abs(mapped)*(1.-1./s))
        else:
            value=mapped
    out[index]=value


def offset_table(reference_frames,source_frames,grasps,layout,hands,profiles,ramp=4.):
    if not 0.<ramp<=10.:raise ValueError('Grip ramp must be in (0, 10] source frames')
    table=np.zeros((reference_frames,len(profiles),layout.action_dim),np.float32)
    source=np.linspace(0,source_frames-1,reference_frames)
    for world,profile in enumerate(profiles):
        if set(profile)!={'left','right'} or any(not np.isfinite(float(v)) or not 0.<=float(v)<=.3 for v in profile.values()):
            raise ValueError('Declare left/right flexion corrections in [0, 0.3] radians')
        for side,start,count in zip(layout.sides,layout.hand_action_starts,layout.finger_counts):
            hand=hands[side]
            assert len(hand.finger_joint_names)==count
            weight=np.zeros(reference_frames)
            for grasp in grasps:
                if grasp['side']!=side:continue
                opening=np.clip((source-(grasp['start']-ramp))/ramp,0.,1.)
                closing=np.clip((grasp['end']+ramp-source)/ramp,0.,1.)
                value=np.minimum(opening,closing)
                weight=np.maximum(weight,value*value*(3.-2.*value))
            for joint,name in enumerate(hand.finger_joint_names):
                if name.endswith(('_FE','_IP','_PIP','_DIP')):
                    table[:,world,start+6+joint]=weight*float(profile[side])
    if not np.isfinite(table).all():raise ValueError('Non-finite finger correction')
    return table


def install():
    raw=os.environ.get('V2D_GRIP_BIAS');sweep=os.environ.get('V2D_GRIP_BIAS_PROFILES')
    if not raw and not sweep:return
    if raw and sweep:raise ValueError('Choose one uniform grip correction or a diagnostic sweep')
    if getattr(BaseEnv,'_declared_grip_bias_installed',False):raise RuntimeError('Grip correction installed twice')
    profiles=json.loads(sweep) if sweep else [json.loads(raw)]
    cfg=json.loads(Path('/episode/episode-config.json').read_text())
    original_init,original_process=BaseEnv.__init__,ResidualHandPoseAction.process
    def initialize(self,*args,**kwargs):
        original_init(self,*args,**kwargs)
        actual=profiles if sweep else profiles*self.world_count
        if len(actual)!=self.world_count:raise ValueError('Each diagnostic world needs exactly one profile')
        buffers=self.action.buffers
        table=offset_table(self.reference.num_frames,cfg['frames'],cfg['grasps'],buffers.layout,
            {h.side:h for h in self.scene.layout.hands},actual)
        buffers._grip_bias_table=wp.array(table,dtype=wp.float32,device=self.device)
        buffers._grip_bias_output=wp.zeros_like(self.action_input)
        self._declared_grip_bias_profiles=actual
    def process(self,action,timestep,state=None):
        b=self.buffers
        wp.launch(apply_bias,dim=action.size,
            inputs=[action,timestep,b._grip_bias_table,b.scale,b.input_scale,b.input_mapping_code,
                    b.reference.num_frames,b.layout.action_dim],outputs=[b._grip_bias_output],device=action.device)
        return original_process(self,b._grip_bias_output,timestep,state)
    BaseEnv.__init__,ResidualHandPoseAction.process=initialize,process
    BaseEnv._declared_grip_bias_installed=True
    print('Declared finger flexion corrections in radians:',profiles,flush=True)
