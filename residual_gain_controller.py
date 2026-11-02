"""Bound learned wrist/finger residuals before the existing reference-relative motor mapping.

The learned actor still supplies every action. This changes only declared robot
action gains, never object forces, object poses, the source clock, or resets.
"""
import json
import os
import numpy as np
import warp as wp
from flash_chord.envs.base import BaseEnv
from flash_chord.runtime.actions.hand_pose import ResidualHandPoseAction


@wp.kernel
def scale_action(action: wp.array(dtype=wp.float32), gains: wp.array2d(dtype=wp.float32), frame:wp.array(dtype=wp.int32), frames:int,
                 action_dim: int, result: wp.array(dtype=wp.float32)):
    index = wp.tid()
    world=index//action_dim;i=wp.clamp(frame[world],0,frames-1)
    result[index] = action[index] * gains[i,index % action_dim]


@wp.kernel
def calibrate_wrist_translation(frame:wp.array(dtype=wp.int32),bias:wp.array2d(dtype=wp.float32),
 position_ids:wp.array(dtype=wp.int32),processed_starts:wp.array(dtype=wp.int32),frames:int,hands:int,
 dofs:int,processed_dim:int,targets:wp.array(dtype=wp.float32),processed:wp.array(dtype=wp.float32)):
 world,hand,axis=wp.tid();i=wp.clamp(frame[world],0,frames-1);delta=bias[i,hand*3+axis]
 targets[world*dofs+position_ids[hand*3+axis]]+=delta
 processed[world*processed_dim+processed_starts[hand]+axis]+=delta


def gains_for_layout(layout, profile):
    gains = np.ones(layout.action_dim, np.float32)
    allowed = {"wrist_position", "wrist_orientation", "finger"}
    optional={'after_source_frame','ramp_source_frames','wrist_position_bias_m','until_source_frame','fade_source_frames'}
    if (not allowed<=set(profile) or not set(profile)<=allowed|optional or any(not 0. <= float(profile[k]) <= 1. for k in allowed)
            or not 0. < float(profile['finger']) <= 1.):
        raise ValueError("Declare wrist gains in [0, 1] and a learned-finger gain in (0, 1]")
    for hand, start in enumerate(layout.hand_action_starts):
        count = layout.finger_counts[hand]
        gains[start:start + 3] = profile["wrist_position"]
        gains[start + 3:start + 6] = profile["wrist_orientation"]
        gains[start + 6:start + 6 + count] = profile["finger"]
    return gains


def source_timed_gains(layout,profile,reference_frames,source_frames):
    gains=gains_for_layout(layout,profile)
    if 'after_source_frame' not in profile:
        if set(profile)&{'ramp_source_frames','until_source_frame','fade_source_frames'}:raise ValueError('A source window requires a start frame')
        return np.broadcast_to(gains,(reference_frames,len(gains))).copy()
    start=float(profile['after_source_frame']);ramp=float(profile.get('ramp_source_frames',8))
    if not 0<=start<source_frames-1 or not 0<ramp<=source_frames-1-start:raise ValueError('Invalid source gain interval')
    weight=np.clip((np.linspace(0,source_frames-1,reference_frames)-start)/ramp,0,1)
    weight=weight*weight*(3-2*weight)
    if 'until_source_frame' in profile:
        end=float(profile['until_source_frame']);fade=float(profile.get('fade_source_frames',4))
        if not start+ramp<=end<source_frames-1 or not 0<fade<=source_frames-1-end:
            raise ValueError('Invalid end of source gain interval')
        release=np.clip((np.linspace(0,source_frames-1,reference_frames)-end)/fade,0,1)
        weight*=1-release*release*(3-2*release)
    elif 'fade_source_frames' in profile:raise ValueError('A fade requires an end frame')
    return (1+weight[:,None]*(gains[None,:]-1)).astype(np.float32)


def install():
    raw = os.environ.get("V2D_RESIDUAL_PROFILE")
    if not raw:
        return
    profile = json.loads(raw)
    if getattr(BaseEnv, "_declared_residual_gains_installed", False):
        raise RuntimeError("Residual gains were installed twice")
    original_init, original_process = BaseEnv.__init__, ResidualHandPoseAction.process

    def initialize(self, *args, **kwargs):
        original_init(self, *args, **kwargs)
        buffers = self.action.buffers
        from pathlib import Path
        cfg=json.loads(Path('/episode/episode-config.json').read_text())
        buffers._declared_residual_gains = wp.array(source_timed_gains(buffers.layout, profile,self.reference.num_frames,cfg['frames']),
                                                   dtype=wp.float32, device=self.device)
        buffers._declared_scaled_input = wp.zeros_like(self.action_input)
        if 'wrist_position_bias_m' in profile:
            values=profile['wrist_position_bias_m'];assert set(values)<=set(buffers.layout.sides)
            offsets=np.zeros((self.reference.num_frames,buffers.layout.num_hands*3),np.float32)
            weight=source_timed_gains(buffers.layout,{**profile,'wrist_position':0.},self.reference.num_frames,cfg['frames'])[:,0]
            weight=1.-weight
            for hand,side in enumerate(buffers.layout.sides):
                vector=np.asarray(values.get(side,[0.,0.,0.]),np.float32)
                assert vector.shape==(3,) and np.isfinite(vector).all() and np.linalg.norm(vector)<=.06
                offsets[:,hand*3:hand*3+3]=weight[:,None]*vector
            buffers._declared_wrist_bias=wp.array(offsets,dtype=wp.float32,device=self.device)

    def process(self, action, timestep, state=None):
        buffers = self.buffers
        wp.launch(scale_action, dim=action.size,
                  inputs=[action, buffers._declared_residual_gains,timestep,buffers.reference.num_frames, buffers.layout.action_dim],
                  outputs=[buffers._declared_scaled_input], device=action.device)
        result=original_process(self, buffers._declared_scaled_input, timestep, state)
        if hasattr(buffers,'_declared_wrist_bias'):
            l=buffers.layout
            wp.launch(calibrate_wrist_translation,dim=(buffers.world_count,l.num_hands,3),inputs=[
                timestep,buffers._declared_wrist_bias,buffers.wrist_pos_dof_ids,buffers.hand_processed_starts,
                buffers.reference.num_frames,l.num_hands,l.num_joint_dof,l.processed_dim,buffers.joint_target,
                buffers.processed_target],device=action.device)
        return result

    BaseEnv.__init__, ResidualHandPoseAction.process = initialize, process
    BaseEnv._declared_residual_gains_installed = True
    print("Declared learned-action residual gains:", profile, flush=True)
