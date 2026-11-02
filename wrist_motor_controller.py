"""Declare a bounded translational wrist motor tuning; object physics is unchanged."""
import copy
import json
import os
import numpy as np
from flash_chord.embodiments.sharpa_hands import SharpaHands,SharpaHandsConfig
from flash_chord.envs.base import BaseEnv


def install():
    raw=os.environ.get('V2D_WRIST_MOTOR_PROFILE')
    if not raw:return
    profile=json.loads(raw)
    if set(profile)!={'kp','kd'} or not 25.<=float(profile['kp'])<=500. or not 6.<=float(profile['kd'])<=40.:
        raise ValueError('Declare wrist stiffness 25..500 and damping 6..40')
    sides=json.loads(os.environ.get('V2D_WRIST_SIDE_MOTOR_PROFILES','{}'))
    for side,value in sides.items():
        if side not in ['left','right'] or set(value)!={'kp','kd'} or not 25.<=float(value['kp'])<=500. or not 6.<=float(value['kd'])<=40.:
            raise ValueError('Invalid per-hand translational motor tuning')
    if getattr(SharpaHands,'_declared_motor_profile',False):raise RuntimeError('Wrist motor profile installed twice')
    original=SharpaHands.__init__;original_env=BaseEnv.__init__
    def initialize(self,config=None):
        value=copy.deepcopy(config or SharpaHandsConfig());overrides=dict(value.joint_overrides)
        overrides[r'wrist_[xyz]$']={**overrides[r'wrist_[xyz]$'],**profile}
        for side,side_profile in sides.items():overrides[side+r'_wrist_[xyz]$']=dict(side_profile)
        value.joint_overrides=overrides
        original(self,value)
    def environment(self,*args,**kwargs):
        original_env(self,*args,**kwargs)
        gains=self.model.joint_target_ke.numpy().reshape(self.world_count,self.num_joint_dof)
        damping=self.model.joint_target_kd.numpy().reshape(self.world_count,self.num_joint_dof)
        limits=self.model.joint_effort_limit.numpy().reshape(self.world_count,self.num_joint_dof)
        for hand in self.scene.layout.hands:
            ids=list(hand.wrist_pos_dof_ids)
            expected=sides.get(hand.side,profile)
            assert np.allclose(gains[:,ids],expected['kp']) and np.allclose(damping[:,ids],expected['kd'])
            assert np.allclose(limits[:,ids],45.),'Wrist actuator force limits must stay unchanged'
    SharpaHands.__init__=initialize;BaseEnv.__init__=environment;SharpaHands._declared_motor_profile=True
    print('Declared translational wrist motor tuning:',profile,'force limits remain 45 N',flush=True)
