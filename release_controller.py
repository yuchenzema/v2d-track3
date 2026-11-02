"""Optional video-timed robot hand release prior; no object forces or state changes."""
import json,os
from pathlib import Path
import numpy as np,warp as wp
from flash_chord.envs.base import BaseEnv
from flash_chord.runtime.actions.hand_pose import ResidualHandPoseAction
from flash_chord.utils.quat import quat_to_rotvec

def selected_finger_columns(names,item):
 prefixes=item.get('finger_joint_prefixes')
 if prefixes is None:return np.arange(len(names))
 assert prefixes and all(isinstance(p,str) and p for p in prefixes)
 assert all(any(n.startswith(p) for n in names) for p in prefixes),'Unknown release finger prefix'
 return np.array([i for i,n in enumerate(names) if n.startswith(tuple(prefixes))],dtype=int)

@wp.kernel
def apply_release(frame:wp.array(dtype=wp.int32),weights:wp.array2d(dtype=wp.float32),reference:wp.array(dtype=wp.float32),dof:wp.array(dtype=wp.int32),processed_ids:wp.array(dtype=wp.int32),action_ids:wp.array(dtype=wp.int32),frames:int,fingers:int,dofs:int,processed_dim:int,action_dim:int,targets:wp.array(dtype=wp.float32),processed:wp.array(dtype=wp.float32),filtered:wp.array(dtype=wp.float32)):
 world,finger=wp.tid();i=wp.clamp(frame[world],0,frames-1);blend=weights[i,finger];target_id=world*dofs+dof[finger];value=(1.-blend)*targets[target_id]+blend*reference[i*fingers+finger];targets[target_id]=value;processed[world*processed_dim+processed_ids[finger]]=value;filtered[world*action_dim+action_ids[finger]]*=1.-blend

@wp.kernel
def apply_terminal_wrist(frame:wp.array(dtype=wp.int32),weights:wp.array2d(dtype=wp.float32),reference_position:wp.array(dtype=wp.vec3),reference_quaternion:wp.array(dtype=wp.quat),position_ids:wp.array(dtype=wp.int32),orientation_ids:wp.array(dtype=wp.int32),processed_starts:wp.array(dtype=wp.int32),action_starts:wp.array(dtype=wp.int32),frames:int,hands:int,dofs:int,processed_dim:int,action_dim:int,targets:wp.array(dtype=wp.float32),processed:wp.array(dtype=wp.float32),filtered:wp.array(dtype=wp.float32)):
 world,hand=wp.tid();i=wp.clamp(frame[world],0,frames-1);blend=weights[i,hand];offset=world*processed_dim+processed_starts[hand];ref=i*hands+hand;position=reference_position[ref]
 current=wp.quat(processed[offset+4],processed[offset+5],processed[offset+6],processed[offset+3]);q=reference_quaternion[ref]
 if wp.dot(current,q)<0.:q=-q
 q=wp.normalize(wp.quat_slerp(current,q,blend));rotation=quat_to_rotvec(q)
 for axis in range(3):
  dof=world*dofs+position_ids[hand*3+axis];value=(1.-blend)*targets[dof]+blend*position[axis];targets[dof]=value;processed[offset+axis]=value;targets[world*dofs+orientation_ids[hand*3+axis]]=rotation[axis]
 processed[offset+3]=q[3];processed[offset+4]=q[0];processed[offset+5]=q[1];processed[offset+6]=q[2]
 for axis in range(6):filtered[world*action_dim+action_starts[hand]+axis]*=1.-blend

def install():
 spec=os.environ.get('V2D_RELEASE_SPEC')
 if not spec:return
 spec=json.loads(spec);cfg=json.loads(Path('/episode/episode-config.json').read_text());original_init=BaseEnv.__init__;original_process=ResidualHandPoseAction.process
 def initialize(self,*args,**kwargs):
  original_init(self,*args,**kwargs);buffers=self.action.buffers;layout=buffers.layout;weights=np.zeros((self.reference.num_frames,layout.total_fingers),np.float32);wrist_weights=np.zeros((self.reference.num_frames,layout.num_hands),np.float32);targets=buffers.reference.finger_joint_pos.numpy().reshape(self.reference.num_frames,layout.total_fingers).copy();source=np.linspace(0,cfg['frames']-1,self.reference.num_frames);processed=[];actions=[]
  for h,side in enumerate(layout.sides):
   start=layout.hand_finger_starts[h];count=layout.finger_counts[h]
   for item in spec:
    if item['side']==side:
     names=self.scene.layout.hand(side).finger_joint_names
     columns=start+selected_finger_columns(names,item)
     new_weights=np.clip((source-item['after_source_frame'])/item.get('ramp_source_frames',2),0.,1.)[:,None]
     weights[:,columns]=np.maximum(weights[:,columns],new_weights)
     if 'open_posture_source_frame' in item:
      index=int(round(item['open_posture_source_frame']*(self.reference.num_frames-1)/(cfg['frames']-1)));targets[:,columns]=targets[index,columns].copy()
     if 'finger_openness_scale' in item:
      openness=float(item['finger_openness_scale']);assert 0.<=openness<=1.
      targets[:,columns]*=openness
     if item.get('zero_wrist_residual'):wrist_weights[:,h]=np.clip((source-item.get('wrist_after_source_frame',item['after_source_frame']))/item.get('ramp_source_frames',2),0.,1.)
   processed.extend(range(layout.hand_processed_starts[h]+7,layout.hand_processed_starts[h]+7+count));actions.extend(range(layout.hand_action_starts[h]+6,layout.hand_action_starts[h]+6+count))
  buffers._video_release_weights=wp.array(weights,dtype=wp.float32,device=self.device);buffers._video_release_targets=wp.array(targets.ravel(),dtype=wp.float32,device=self.device);buffers._video_release_processed=wp.array(processed,dtype=wp.int32,device=self.device);buffers._video_release_actions=wp.array(actions,dtype=wp.int32,device=self.device)
  buffers._video_release_wrist_weights=wp.array(wrist_weights,dtype=wp.float32,device=self.device)
 def process(self,action,timestep,state=None):
  original_process(self,action,timestep,state);b=self.buffers
  if not hasattr(b,'_video_release_weights'):return
  l=b.layout;wp.launch(apply_release,dim=(b.world_count,l.total_fingers),inputs=[timestep,b._video_release_weights,b._video_release_targets,b.finger_dof_ids,b._video_release_processed,b._video_release_actions,b.reference.num_frames,l.total_fingers,l.num_joint_dof,l.processed_dim,l.action_dim],outputs=[b.joint_target,b.processed_target,b.filtered_action],device=b.joint_target.device)
  wp.launch(apply_terminal_wrist,dim=(b.world_count,l.num_hands),inputs=[timestep,b._video_release_wrist_weights,b.reference.wrist_pos_w,b.reference.wrist_quat_w,b.wrist_pos_dof_ids,b.wrist_ori_dof_ids,b.hand_processed_starts,b.hand_action_starts,b.reference.num_frames,l.num_hands,l.num_joint_dof,l.processed_dim,l.action_dim],outputs=[b.joint_target,b.processed_target,b.filtered_action],device=b.joint_target.device)
 BaseEnv.__init__=initialize;ResidualHandPoseAction.process=process
