"""Tighten robot wrist reference tracking while preserving actor residual spring forces.

Only robot motor stiffness/damping and wrist translation residual gains change.
Objects, force limits, reference timing and reset rules are unchanged.
"""
import json,os
import numpy as np,warp as wp
from flash_chord.envs.base import BaseEnv
from flash_chord.runtime.actions.hand_pose import ResidualHandPoseAction

@wp.kernel
def rescale_residual(action:wp.array(dtype=wp.float32),gains:wp.array(dtype=wp.float32),
        input_scale:wp.array(dtype=wp.float32),mapping:int,dimension:int,out:wp.array(dtype=wp.float32)):
 index=wp.tid();component=index%dimension;gain=gains[component];value=action[index]
 if gain!=1.:
  s=input_scale[component];mapped=value
  if mapping==1:mapped=s*wp.clamp(value,-1.,1.)
  elif mapping==2:
   a=wp.clamp(value,-1.,1.);mapped=a/((1.-wp.abs(a))+wp.abs(a)/s)
  mapped*=gain
  if mapping==1:value=wp.clamp(mapped/s,-1.,1.)
  elif mapping==2:value=mapped/(1.+wp.abs(mapped)*(1.-1./s))
  else:value=mapped
 out[index]=value

def gains_for_layout(layout,profile):
 gains=np.ones(layout.action_dim,np.float32)
 for side,value in profile.items():
  assert side in layout.sides and set(value)=={'kp','kd','original_kp'}
  assert 25.<=float(value['original_kp'])<=float(value['kp'])<=500. and 6.<=float(value['kd'])<=40.
  hand=layout.sides.index(side);start=layout.hand_action_starts[hand];gains[start:start+3]=float(value['original_kp'])/float(value['kp'])
 return gains

def install():
 profile=json.loads(os.environ['V2D_WRIST_IMPEDANCE_PROFILE']);assert profile and set(profile)<= {'left','right'}
 assert not getattr(BaseEnv,'_impedance_profile_installed',False)
 os.environ['V2D_WRIST_MOTOR_PROFILE']=json.dumps({'kp':25.,'kd':6.})
 os.environ['V2D_WRIST_SIDE_MOTOR_PROFILES']=json.dumps({side:{'kp':v['kp'],'kd':v['kd']} for side,v in profile.items()})
 from wrist_motor_controller import install as motors
 motors();initialize=BaseEnv.__init__;process_action=ResidualHandPoseAction.process
 def init(self,*args,**kwargs):
  initialize(self,*args,**kwargs);b=self.action.buffers;gains=gains_for_layout(b.layout,profile)
  b._impedance_residual_gains=wp.array(gains,dtype=wp.float32,device=self.device);b._impedance_scaled_action=wp.zeros_like(self.action_input)
 def process(self,action,timestep,state=None):
  b=self.buffers
  wp.launch(rescale_residual,dim=action.size,inputs=[action,b._impedance_residual_gains,b.input_scale,b.input_mapping_code,b.layout.action_dim],outputs=[b._impedance_scaled_action],device=action.device)
  return process_action(self,b._impedance_scaled_action,timestep,state)
 BaseEnv.__init__=init;ResidualHandPoseAction.process=process;BaseEnv._impedance_profile_installed=True
 print('Declared wrist impedance with force-preserving positional residual scales:',profile,flush=True)
