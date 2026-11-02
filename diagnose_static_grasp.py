"""Diagnostic: hold reconstructed grasps still, then lift the robot hands 3 cm.

This is not a learned-policy or source-timed rollout. Only hand motor targets
move; object physics remains unassisted and is never reset during the probe.
"""
import json
import os
from pathlib import Path
import runpy
import sys
import hydra
import numpy as np
from omegaconf import OmegaConf,open_dict
from scipy.spatial.transform import Rotation
import warp as wp
import flash_chord.training.flash_sac.runner as runner
from flash_chord.evaluation.harness import apply_evaluation_composition

ROOT=Path('/episode');cfg=json.loads((ROOT/'episode-config.json').read_text())
probe_reference=os.environ.get('V2D_GRASP_PROBE_REFERENCE',cfg['training_reference'])
probe_world_count=int(os.environ.get('V2D_STATIC_GRASP_WORLD_COUNT','1'))
source_frame=min(cfg['frames']-1,max(g['start'] for g in cfg['grasps'])+12)
if os.environ.get('V2D_STATIC_GRASP_SOURCE_FRAME'):
    source_frame=int(os.environ['V2D_STATIC_GRASP_SOURCE_FRAME'])
    assert 0<=source_frame<cfg['frames']
motor_profile=json.loads(os.environ['V2D_WRIST_MOTOR_PROFILE']) if os.environ.get('V2D_WRIST_MOTOR_PROFILE') else None
side_profiles=json.loads(os.environ.get('V2D_WRIST_SIDE_MOTOR_PROFILES','{}'))
lift_sides=json.loads(os.environ.get('V2D_STATIC_GRASP_LIFT_SIDES','["left","right"]'))
assert lift_sides and set(lift_sides)<= {'left','right'}
for side,value in side_profiles.items():
    assert side in ['left','right'] and set(value)=={'kp','kd'} and 25.<=float(value['kp'])<=500. and 6.<=float(value['kd'])<=40.
if motor_profile and (set(motor_profile)!={'kp','kd'} or not 25.<=float(motor_profile['kp'])<=500. or not 6.<=float(motor_profile['kd'])<=40.):
    raise ValueError('Probe translational stiffness must be 25..500 N/m with damping 6..40')
suffix=f"-kp{motor_profile['kp']:g}-kd{motor_profile['kd']:g}" if motor_profile else ''
label=f'static-grasp-hold-lift-source-{source_frame}-v2{suffix}'+('-reference-'+probe_reference if probe_reference!=cfg['training_reference'] else '')
if set(lift_sides)!={'left','right'}:label+='-lift-'+'-'.join(sorted(lift_sides))


def probe(env,*args,**kwargs):
    e=env.env
    assert not e.config.auto_reset
    frame=int(round(source_frame*(e.reference.num_frames-1)/(cfg['frames']-1)))
    e.set_voc_scale(0.);e.reset(frame_id=frame);e.set_voc_scale(0.)
    initial=e.state_0.body_q.numpy().copy();names=list(e.model.body_label)
    layout=e.action.buffers.layout;scale=e.action.buffers.scale.numpy()
    poses=[];commands=[]
    for step in range(60):
        # Pin only the hand-command clock for this explicitly static diagnostic.
        # No object state or force is written here.
        e.timestep.fill_(frame)
        lift=.03*float(np.clip((step-19)/20.,0.,1.))
        raw=np.zeros(e.action_input.size,np.float32)
        for side,start in zip(layout.sides,layout.hand_action_starts):
            if side in lift_sides:raw[start+2]=lift/scale[start+2]
        e.step(wp.array(raw,device=e.device))
        assert np.all(e.voc_scale.numpy()==0.)
        poses.append(e.state_0.body_q.numpy().copy());commands.append(lift)
    states=np.asarray(poses)
    assert np.isfinite(states).all()
    objects={}
    for body,name in enumerate(cfg['objects']):
        grasps=[g for g in cfg['grasps'] if g['object']==body and g['start']<=source_frame<=g['end']]
        if not grasps:
            object_id=names.index(f'track3_{name}/object');actual=states[:,object_id];p0=initial[object_id]
            drift=np.linalg.norm(actual[:,:3]-p0[:3],axis=1)
            angles=(Rotation.from_quat(p0[3:]).inv()*Rotation.from_quat(actual[:,3:])).magnitude()
            objects[name]={'side':None,'no_active_video_grasp':True,'stable_hold_and_lift':False,
                'source_object_marked_stationary':body in cfg.get('observed_stationary_objects',[]),
                'maximum_passive_position_drift_m':float(drift.max()),'maximum_passive_rotation_drift_rad':float(angles.max()),
                'passive_rest_stable':bool(drift.max()<=.01 and angles.max()<=.2)}
            continue
        if len(grasps)!=1:raise ValueError('Probe requires one active grasp for each object')
        side=grasps[0]['side'];object_id=names.index(f'track3_{name}/object')
        wrist_id=names.index(f'{side}_sharpa_wave/{side}_hand_C_MC')
        actual=states[:,object_id];p0=initial[object_id]
        # Measure the commanded lift after the static settling interval. Keep
        # initial load settling as its own check instead of counting it twice.
        wrist_lift=states[-1,wrist_id,:3]-states[19,wrist_id,:3]
        static_drift=float(np.linalg.norm(actual[19,:3]-p0[:3]))
        follow_error=float(np.linalg.norm(actual[-1,:3]-(actual[19,:3]+wrist_lift)))
        angle=float((Rotation.from_quat(p0[3:]).inv()*Rotation.from_quat(actual[-1,3:])).magnitude())
        objects[name]={'side':side,'static_position_drift_m':static_drift,
            'actual_hand_translation_m':wrist_lift.tolist(),
            'static_hand_translation_m':(states[19,wrist_id,:3]-initial[wrist_id,:3]).tolist(),
            'actual_object_translation_m':(actual[-1,:3]-actual[19,:3]).tolist(),
            'object_following_error_m':follow_error,'object_orientation_change_rad':angle,
            'stable_hold_and_lift':bool(static_drift<=.015 and follow_error<=.02 and angle<=.3 and wrist_lift[2]>=.02),
            'source_object_marked_stationary':body in cfg.get('observed_stationary_objects',[])}
    gains=e.model.joint_target_ke.numpy();damping=e.model.joint_target_kd.numpy();limits=e.model.joint_effort_limit.numpy()
    motor_parameters={h.side:{'stiffness_N_per_m':gains[list(h.wrist_pos_dof_ids)].tolist(),
        'damping_Ns_per_m':damping[list(h.wrist_pos_dof_ids)].tolist(),
        'force_limit_N':limits[list(h.wrist_pos_dof_ids)].tolist()} for h in e.scene.layout.hands}
    if motor_profile:
        assert all(np.allclose(v['stiffness_N_per_m'],side_profiles.get(side,motor_profile)['kp']) and
            np.allclose(v['damping_Ns_per_m'],side_profiles.get(side,motor_profile)['kd']) and
            np.allclose(v['force_limit_N'],45.) for side,v in motor_parameters.items())
    report={'episode':cfg['episode'],'method':__doc__,'diagnostic_protocol_version':2,'diagnostic_only':True,'policy_inference':False,
        'wrist_motor_parameters':motor_parameters,'wrist_side_motor_profiles':side_profiles,'reference':probe_reference,
        'frozen_source_frame':source_frame,'frozen_native_reference_frame':frame,
        'static_control_steps':20,'lifting_control_steps':20,'holding_control_steps':20,
        'commanded_hand_lift_m':.03,'lifted_robot_sides':lift_sides,'object_assistance':0.,'object_state_resets_during_probe':False,
        'objects':objects,'full_human_motion_match_verified':False,
        'limitation':'A static/lift test does not verify the approach or full source motion; lifting a source-stationary support is a stronger test than its original task.'}
    (ROOT/f'evaluation/{label}.json').write_text(json.dumps(report,indent=2))
    np.savez_compressed(ROOT/f'evaluation/{label}.npz',body_poses=states,initial_body_poses=initial,
        body_names=names,commanded_lift_m=commands,diagnostic_kind=__doc__,frozen_reference_frame=frame)
    print(json.dumps(report,indent=2),flush=True)
    raise SystemExit(0)


runner.run_training=probe
sys.argv=[sys.argv[0],'experiment=sharpa_flash_sac',
    f"task.parquet='/episode/human_motion_data/{probe_reference}/processed/sequence_id={cfg['sequence']}/robot_name=sharpa_wave'",
    'task.motion_speed=0.5',f"sim.substeps={cfg.get('physics_substeps',5)}",f'scene.world_count={probe_world_count}',
    'action.input_mode=raw','sim.njmax=1536','reset.reset_finger_openness=0.0','logging.mode=disabled',
    f'training.total_environment_steps={((250_003_456+probe_world_count-1)//probe_world_count)*probe_world_count}',
    'output_dir=/episode/diagnostic-checkpoints','hydra.run.dir=/episode/hydra/static-grasp-probe']
native=runpy.run_path('/workspace/scripts/train_flash_sac.py',run_name='native_training')


@hydra.main(version_base='1.3',config_path='/workspace/src/flash_chord/configs',config_name='train_flash_sac')
def main(config):
    if os.environ.get('V2D_DIAGNOSTIC_WRIST_ANGULAR_KP'):
        angular=float(os.environ['V2D_DIAGNOSTIC_WRIST_ANGULAR_KP'])
        assert 25.<=angular<=250.
        config.action.wrist_kp=angular
    if motor_profile:
        from flash_chord.embodiments.sharpa_hands import SharpaHandsConfig
        overrides=dict(SharpaHandsConfig().joint_overrides)
        overrides[r'wrist_[xyz]$']={**overrides[r'wrist_[xyz]$'],**motor_profile}
        for side,value in side_profiles.items():overrides[side+r'_wrist_[xyz]$']=dict(value)
        with open_dict(config.embodiment.config):config.embodiment.config.joint_overrides=overrides
    config=OmegaConf.create(apply_evaluation_composition(OmegaConf.to_container(config,resolve=True),world_count=probe_world_count,reset_mode='explicit'))
    native['main'].__wrapped__(config)


if __name__=='__main__':main()
