"""Source-timed robot wrist impedance, applied at every physics substep.

The native25N/m motor receives an equivalent position command, preserving
the actor's residual spring force and its existing45N actuator limit.
Only wrist joint motor commands change; no object state or force is accessed.
"""
import json,os
from pathlib import Path
import numpy as np,warp as wp
from flash_chord.envs.base import BaseEnv
from flash_chord.runtime.actions.hand_pose import ResidualHandPoseAction

@wp.kernel
def impedance_targets(frame:wp.array(dtype=wp.int32),kp:wp.array2d(dtype=wp.float32),kd:wp.array2d(dtype=wp.float32),
 reference:wp.array(dtype=wp.vec3),q:wp.array(dtype=wp.float32),qd:wp.array(dtype=wp.float32),
 q_ids:wp.array(dtype=wp.int32),dof_ids:wp.array(dtype=wp.int32),native_target:wp.array(dtype=wp.float32),
 frames:int,hands:int,numq:int,numdof:int,out:wp.array(dtype=wp.float32)):
 world,hand,axis=wp.tid();i=wp.clamp(frame[world],0,frames-1);d=world*numdof+dof_ids[hand*3+axis]
 position=q[world*numq+q_ids[hand*3+axis]];velocity=qd[d];target=reference[i*hands+hand][axis]
 out[d]=native_target[d]+(kp[i,hand*3+axis]/25.-1.)*(target-position)-(kd[i,hand*3+axis]-6.)/25.*velocity

def tables(profile,sides,native_frames,source_frames):
 assert profile and set(profile)<=set(sides)
 kp=np.full((native_frames,len(sides)*3),25.,np.float32);kd=np.full_like(kp,6.)
 clock=np.linspace(0,source_frames-1,native_frames)
 for side,spec in profile.items():
  required={'kp','kd','start_source_frame','ramp_source_frames'}
  assert required<=set(spec) and set(spec)<=required|{'axes','until_source_frame','fade_source_frames'}
  assert 25<=spec['kp']<=250 and 6<=spec['kd']<=20
  start=float(spec['start_source_frame']);ramp=float(spec['ramp_source_frames'])
  assert 0<=start<source_frames-1 and 0<ramp<=source_frames-1-start
  w=np.clip((clock-start)/ramp,0,1);w=w*w*(3-2*w);h=sides.index(side)
  if 'until_source_frame' in spec:
   end=float(spec['until_source_frame']);fade=float(spec.get('fade_source_frames',4))
   assert start+ramp<=end<source_frames-1 and 0<fade<=source_frames-1-end
   release=np.clip((clock-end)/fade,0,1);w*=1-release*release*(3-2*release)
  else:assert 'fade_source_frames' not in spec
  axes=spec.get('axes',['x','y','z']);assert axes and len(set(axes))==len(axes) and set(axes)<=set('xyz')
  for axis in axes:
   k=h*3+'xyz'.index(axis);kp[:,k]=25+(spec['kp']-25)*w;kd[:,k]=6+(spec['kd']-6)*w
 return kp,kd

def install():
 profile=json.loads(os.environ['V2D_TIMED_WRIST_IMPEDANCE_PROFILE'])
 assert not getattr(BaseEnv,'_timed_impedance_installed',False)
 original_init=BaseEnv.__init__;original_apply=ResidualHandPoseAction.apply_control
 def initialize(self,*args,**kwargs):
  original_init(self,*args,**kwargs);b=self.action.buffers;l=b.layout
  cfg=json.loads(Path('/episode/episode-config.json').read_text());kp,kd=tables(profile,list(l.sides),self.reference.num_frames,cfg['frames'])
  b._timed_kp=wp.array(kp,dtype=wp.float32,device=self.device);b._timed_kd=wp.array(kd,dtype=wp.float32,device=self.device)
  b._timed_clock=self.timestep;b._timed_num_q=self.model.joint_coord_count//self.world_count
  ids=[]
  for side in l.sides:
   hand=next(h for h in self.scene.layout.hands if h.side==side);ids.extend(hand.wrist_pos_q_ids)
   dofs=list(hand.wrist_pos_dof_ids)
   for name,value in [('joint_target_ke',25.),('joint_target_kd',6.),('joint_effort_limit',45.)]:
    a=getattr(self.model,name).numpy().reshape(self.world_count,self.num_joint_dof);assert np.allclose(a[:,dofs],value),(side,name)
  b._timed_q_ids=wp.array(ids,dtype=wp.int32,device=self.device)
 def apply(self,state,control):
  original_apply(self,state,control);b=self.buffers;l=b.layout
  wp.launch(impedance_targets,dim=(b.world_count,l.num_hands,3),inputs=[b._timed_clock,b._timed_kp,b._timed_kd,
   b.reference.wrist_pos_w,state.joint_q,state.joint_qd,b._timed_q_ids,b.wrist_pos_dof_ids,b.joint_target,
   b.reference.num_frames,l.num_hands,b._timed_num_q,l.num_joint_dof,control.joint_target_pos],device=state.joint_q.device)
 BaseEnv.__init__=initialize;ResidualHandPoseAction.apply_control=apply;BaseEnv._timed_impedance_installed=True
 print('Declared source-timed wrist impedance at physics substeps; native45N limits retained:',profile,flush=True)
