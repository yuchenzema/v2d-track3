"""Diagnostic robot wrist calibration grid; never selects a world for publication."""
import json,os
from pathlib import Path
import numpy as np,warp as wp
from flash_chord.envs.base import BaseEnv
from flash_chord.runtime.actions.hand_pose import ResidualHandPoseAction


@wp.kernel
def add_bias(frame:wp.array(dtype=wp.int32),values:wp.array3d(dtype=wp.float32),
             position_ids:wp.array(dtype=wp.int32),starts:wp.array(dtype=wp.int32),
             frames:int,dofs:int,processed_dim:int,targets:wp.array(dtype=wp.float32),processed:wp.array(dtype=wp.float32)):
    world,hand,axis=wp.tid();i=wp.clamp(frame[world],0,frames-1);delta=values[world,i,hand*3+axis]
    targets[world*dofs+position_ids[hand*3+axis]]+=delta
    processed[world*processed_dim+starts[hand]+axis]+=delta


def install():
    profiles=json.loads(os.environ['V2D_WRIST_BIAS_PROFILES'])
    original_init,original_process=BaseEnv.__init__,ResidualHandPoseAction.process
    def initialize(self,*args,**kwargs):
        original_init(self,*args,**kwargs);assert self.world_count==len(profiles)
        cfg=json.loads(Path('/episode/episode-config.json').read_text());l=self.action.buffers.layout
        clock=np.linspace(0,cfg['frames']-1,self.reference.num_frames);values=np.zeros((self.world_count,self.reference.num_frames,l.num_hands*3),np.float32)
        for world,profile in enumerate(profiles):
            for side,vector in profile.items():
                assert side in l.sides;vector=np.asarray(vector,np.float32);assert vector.shape==(3,) and np.isfinite(vector).all() and np.linalg.norm(vector)<=.02
                grasp=next(g for g in cfg['grasps'] if g['side']==side)
                onset=max(0,grasp['start']-4);ramp=min(8,cfg['frames']-onset-1)
                a=np.clip((clock-onset)/ramp,0,1);a=a*a*(3-2*a)
                fade=np.clip((clock-grasp['end'])/max(1,min(4,cfg['frames']-1-grasp['end'])),0,1);a*=1-fade*fade*(3-2*fade)
                h=l.sides.index(side);values[world,:,h*3:h*3+3]=a[:,None]*vector
        self.action.buffers._diagnostic_wrist_bias=wp.array(values,dtype=wp.float32,device=self.device)
    def process(self,action,timestep,state=None):
        result=original_process(self,action,timestep,state);b=self.buffers;l=b.layout
        wp.launch(add_bias,dim=(b.world_count,l.num_hands,3),inputs=[timestep,b._diagnostic_wrist_bias,b.wrist_pos_dof_ids,b.hand_processed_starts,b.reference.num_frames,l.num_joint_dof,l.processed_dim,b.joint_target,b.processed_target],device=action.device)
        return result
    BaseEnv.__init__,ResidualHandPoseAction.process=initialize,process
