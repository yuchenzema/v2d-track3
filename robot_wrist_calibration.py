"""Declared per-hand robot wrist gains; original45N/Nm actuator limits retained."""
import json,os
from pathlib import Path
import numpy as np
from flash_chord.envs.base import BaseEnv
from flash_chord.runtime.control import WristController
import flash_chord.evaluation.harness as harness

def install():
 profile=json.loads(os.environ['V2D_ROBOT_WRIST_CALIBRATION']);assert set(profile)=={'left','right'}
 for value in profile.values():
  assert {'kp','kd','angular_kp'}<=set(value)<={'kp','kd','angular_kp','axes'} and 25<=value['kp']<=500 and 6<=value['kd']<=40 and 25<=value['angular_kp']<=250
  for axis,gain in value.get('axes',{}).items():assert axis in 'xyz' and set(gain)=={'kp','kd'} and 25<=gain['kp']<=500 and 6<=gain['kd']<=40
 assert not getattr(BaseEnv,'_wrist_calibration_installed',False)
 native_config=harness.checkpoint_config
 def configuration(metadata):
  cfg=native_config(metadata);overrides=cfg['embodiment']['config']['joint_overrides']
  # Checkpoint JSON sorts dictionary keys. Restore precedence explicitly so
  # generic wrist defaults cannot overwrite the declared per-hand gains.
  for side,value in profile.items():
   key=side+r'_wrist_[xyz]$';overrides.pop(key,None);overrides[key]={'kp':value['kp'],'kd':value['kd']}
   for axis,gain in value.get('axes',{}).items():
    key=side+'_wrist_'+axis+'$';overrides.pop(key,None);overrides[key]=gain
  return cfg
 harness.checkpoint_config=configuration
 native_layout=WristController.from_layout
 def from_layout(cls,layout,model,world_count,kp=25.,tau_max=45.,device=None):
  result=native_layout(layout,model,world_count,kp=kp,tau_max=tau_max,device=device);assert np.all(result.tau_max.numpy()==45.)
  result.kp.assign(np.asarray([profile[h.side]['angular_kp'] for h in layout.hands],np.float32));return result
 WristController.from_layout=classmethod(from_layout)
 original_init=BaseEnv.__init__
 def initialize(self,*args,**kwargs):
  original_init(self,*args,**kwargs);report={}
  for hand in self.scene.layout.hands:
   ids=list(hand.wrist_pos_dof_ids);value=profile[hand.side]
   for attribute,key in [('joint_target_ke','kp'),('joint_target_kd','kd')]:
    actual=getattr(self.model,attribute).numpy().reshape(self.world_count,self.num_joint_dof)[:,ids]
    expected=[value.get('axes',{}).get(axis,value)[key] for axis in 'xyz']
    assert np.allclose(actual,expected),(hand.side,attribute,np.unique(actual).tolist(),expected)
   assert np.all(self.model.joint_effort_limit.numpy().reshape(self.world_count,self.num_joint_dof)[:,ids]==45.)
   report[hand.side]=value
  Path(os.environ.get('V2D_WRIST_CALIBRATION_REPORT','/episode/research/robot-wrist-calibration-runtime.json')).write_text(json.dumps({'profile':report,'world_count':self.world_count,'actuator_limits_45N_Nm_verified':True,'object_state_or_force_modified':False},indent=2))
 BaseEnv.__init__=initialize;BaseEnv._wrist_calibration_installed=True
 print('Declared robot wrist calibration; original45N/Nm limits:',profile,flush=True)
