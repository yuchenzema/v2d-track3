"""Bounded live object tracking feedback through robot wrist motors only.

The existing trained actor remains active. Corrections affect only wrist motor
targets during the observed grasp; they never write object state or forces.
Diagnostic world grids must be revalidated uniformly before publication.
"""
import json,os
from pathlib import Path
import numpy as np
import warp as wp
from flash_chord.envs.base import BaseEnv
from flash_chord.runtime.actions.hand_pose import ResidualHandPoseAction


@wp.kernel
def scale_robot_action(action:wp.array(dtype=wp.float32),gains:wp.array2d(dtype=wp.float32),dimension:int,result:wp.array(dtype=wp.float32)):
    index=wp.tid();result[index]=action[index]*gains[index//dimension,index%dimension]


@wp.kernel
def correct(frame:wp.array(dtype=wp.int32),gains:wp.array3d(dtype=wp.float32),limits:wp.array(dtype=wp.float32),
            hand_objects:wp.array(dtype=wp.int32),object_bodies:wp.array(dtype=wp.int32),
            reference_bodies:wp.array(dtype=wp.int32),reference:wp.array(dtype=wp.vec3),offsets:wp.array(dtype=wp.vec3),
            body_pose:wp.array(dtype=wp.transform),position_ids:wp.array(dtype=wp.int32),starts:wp.array(dtype=wp.int32),
            frames:int,objects:int,ref_bodies:int,dofs:int,processed_dim:int,
            targets:wp.array(dtype=wp.float32),processed:wp.array(dtype=wp.float32)):
    world,hand=wp.tid();i=wp.clamp(frame[world],0,frames-1);obj=hand_objects[hand]
    desired=reference[i*ref_bodies+reference_bodies[obj]]+offsets[world*objects+obj]
    actual=wp.transform_get_translation(body_pose[object_bodies[world*objects+obj]])
    delta=(desired-actual)*gains[world,i,hand]
    length=wp.length(delta);cap=limits[world]
    if length>cap:delta*=cap/wp.max(length,1.e-8)
    for axis in range(3):
        targets[world*dofs+position_ids[hand*3+axis]]+=delta[axis]
        processed[world*processed_dim+starts[hand]+axis]+=delta[axis]


def install():
    raw=os.environ.get('V2D_ROBOT_TRACKING_FEEDBACK');sweep=os.environ.get('V2D_ROBOT_TRACKING_PROFILES')
    if not raw and not sweep:return
    if raw and sweep:raise ValueError('Use one uniform profile or a diagnostic grid')
    profiles=json.loads(sweep) if sweep else [json.loads(raw)]
    for p in profiles:
        assert {'left','right','maximum_correction_m'}<=set(p)<= {'left','right','maximum_correction_m','release_lead_source_frames','right_wrist_residual_gain','right_finger_residual_gain','left_wrist_position_residual_gains'}
        assert all(np.isfinite(p[k]) and 0<=p[k]<=4 for k in ['left','right'])
        assert .001<=p['maximum_correction_m']<=.03
        assert 0<=p.get('release_lead_source_frames',0)<=20
        assert all(0<=p.get(k,1.)<=1. for k in ['right_wrist_residual_gain','right_finger_residual_gain'])
        if 'left_wrist_position_residual_gains' in p:assert len(p['left_wrist_position_residual_gains'])==3 and all(np.isfinite(v) and 0<=v<=1 for v in p['left_wrist_position_residual_gains'])
    assert not getattr(BaseEnv,'_robot_tracking_feedback_installed',False)
    original_init,original_process=BaseEnv.__init__,ResidualHandPoseAction.process
    def initialize(self,*args,**kwargs):
        original_init(self,*args,**kwargs);b=self.action.buffers;l=b.layout;c=self.command
        active=profiles if sweep else profiles*self.world_count;assert len(active)==self.world_count
        cfg=json.loads(Path('/episode/episode-config.json').read_text())
        source=np.linspace(0,cfg['frames']-1,self.reference.num_frames)
        table=np.zeros((self.world_count,self.reference.num_frames,l.num_hands),np.float32);hand_objects=[]
        for hand,side in enumerate(l.sides):
            grasp=next(g for g in cfg['grasps'] if g['side']==side);hand_objects.append(grasp['object'])
            onset=np.clip((source-grasp['start'])/4.,0,1)
            for world,p in enumerate(active):
                release=np.clip((grasp['end']-p.get('release_lead_source_frames',0)-source)/4.,0,1)
                weight=np.minimum(onset,release);weight=weight*weight*(3-2*weight)
                table[world,:,hand]=weight*p[side]
        b._tracking_gains=wp.array(table,dtype=wp.float32,device=self.device)
        b._tracking_limits=wp.array([p['maximum_correction_m'] for p in active],dtype=wp.float32,device=self.device)
        b._tracking_objects=wp.array(hand_objects,dtype=wp.int32,device=self.device);b._tracking_command=c
        action_gains=np.ones((self.world_count,l.action_dim),np.float32)
        h=l.sides.index('right');start=l.hand_action_starts[h];count=l.finger_counts[h]
        for world,p in enumerate(active):
            action_gains[world,start:start+6]=p.get('right_wrist_residual_gain',1.)
            action_gains[world,start+6:start+6+count]=p.get('right_finger_residual_gain',1.)
            left=l.hand_action_starts[l.sides.index('left')]
            action_gains[world,left:left+3]=p.get('left_wrist_position_residual_gains',[1.,1.,1.])
        b._tracking_action_gains=wp.array(action_gains,dtype=wp.float32,device=self.device)
        b._tracking_scaled_action=wp.zeros_like(self.action_input)
        assert np.all(self.object_control_scale.numpy()==0.)
    def process(self,action,timestep,state=None):
        b=self.buffers
        wp.launch(scale_robot_action,dim=action.size,inputs=[action,b._tracking_action_gains,b.layout.action_dim,b._tracking_scaled_action],device=action.device)
        original_process(self,b._tracking_scaled_action,timestep,state);assert state is not None
        b=self.buffers;l=b.layout;c=b._tracking_command
        wp.launch(correct,dim=(b.world_count,l.num_hands),inputs=[timestep,b._tracking_gains,b._tracking_limits,b._tracking_objects,
            c.voc_body_ids_w,c.voc_reference_body_ids,c.reference.body_pos_w,c.object_root_position_offsets_w,state.body_q,
            b.wrist_pos_dof_ids,b.hand_processed_starts,b.reference.num_frames,c.layout.num_objects,c.reference.num_bodies,
            l.num_joint_dof,l.processed_dim,b.joint_target,b.processed_target],device=action.device)
    BaseEnv.__init__,ResidualHandPoseAction.process=initialize,process
    BaseEnv._robot_tracking_feedback_installed=True
    print('Robot wrist tracking feedback only; bounded motor targets, zero object assistance:',profiles,flush=True)
