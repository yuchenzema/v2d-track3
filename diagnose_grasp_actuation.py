"""Serial unassisted hand-motor diagnostics; frozen source inputs stay unchanged.

Compare zero residual with gravity-load compensation on the robot motors only.
The probe never writes object forces, reference files, checkpoint weights or
object states after the initial reset of each separately reported trial.
"""
import hashlib
import json
import os
from pathlib import Path
import runpy
import sys

import hydra
import mujoco
import numpy as np
import warp as wp
from omegaconf import OmegaConf, open_dict
from scipy.spatial.transform import Rotation
import flash_chord.training.flash_sac.runner as runner
from flash_chord.evaluation.harness import apply_evaluation_composition

sys.path.insert(0, '/tmp/actuation-probe')
sys.path.insert(1, '/episode/scripts')
from grasp_physics_checks import gravity_cone_solution, native_body_id
from native_pinch_contacts import NativePinchContacts
from legacy_source_frame_playback import install as install_source_frame_playback
install_source_frame_playback()

ROOT=Path('/episode')
CFG=json.loads((ROOT/'episode-config.json').read_text())
SOURCE_FRAME=int(os.environ.get('V2D_ACTUATION_FRAME','50'))
OUTPUT=Path(os.environ.get('V2D_ACTUATION_OUTPUT','/tmp/actuation-probe/result.json'))


def probe(env,*args,**kwargs):
    e=env.env;m=e.solver.mj_model;d=e.solver.mj_data
    assert not e.config.auto_reset and e.reference.num_frames==CFG['frames']
    assert e.config.voc.max_force==0. and e.config.voc.max_torque==0.
    grasp=next(g for g in CFG['grasps'] if g['start']<=SOURCE_FRAME<=g['end'])
    side=grasp['side'];name=CFG['objects'][grasp['object']]
    object_id=native_body_id(lambda label:mujoco.mj_name2id(m,mujoco.mjtObj.mjOBJ_BODY,label),f'track3_{name}/object')
    e.reset(frame_id=SOURCE_FRAME);e.set_voc_scale(0.)
    e.solver._update_mjc_data(d,e.model,e.state_0);mujoco.mj_forward(m,d)
    labels=[mujoco.mj_id2name(m,mujoco.mjtObj.mjOBJ_BODY,i) or '' for i in range(m.nbody)]
    hand_bodies=[i for i,n in enumerate(labels) if n.startswith(side+'_sharpa_wave_')]
    hand_geoms=np.flatnonzero(np.isin(m.geom_bodyid,hand_bodies))
    pinch=NativePinchContacts(m,d,hand_geoms,object_id,np.ones(5,dtype=bool))
    contacts=[c for c in pinch.query() if c['valid_witness'] and np.linalg.norm(c['normal_on_object'])>.5]
    solution=gravity_cone_solution([c['normal_on_object'] for c in contacts],[c['position'] for c in contacts],
                                  [c['friction'] for c in contacts],m.body_mass[object_id],d.xipos[object_id],gravity=m.opt.gravity)
    torque=np.zeros(m.nv)
    if solution:
        for c,force in zip(contacts,solution['forces_N']):
            jac=np.zeros((3,m.nv));mujoco.mj_jac(m,d,jac,None,np.array(c['position']),int(m.geom_bodyid[c['hand_geom']]))
            torque+=jac.T@force
    row=__import__('pyarrow.parquet',fromlist=['ParquetFile']).ParquetFile(next((ROOT/f'human_motion_data/{CFG["training_reference"]}/processed').rglob('motion.parquet'))).read().to_pylist()[0]
    finger_names=row[f'{side}_robot_finger_joint_names']
    joints=[mujoco.mj_name2id(m,mujoco.mjtObj.mjOBJ_JOINT,side+'_sharpa_wave_'+n) for n in finger_names]
    assert min(joints)>=0
    finger_dofs=np.array([m.jnt_dofadr[j] for j in joints]);finger_q=np.array([m.jnt_qposadr[j] for j in joints])
    hand=next(h for h in e.scene.layout.hands if h.side==side)
    gains=e.model.joint_target_ke.numpy();kp=gains[list(hand.finger_dof_ids)]
    desired=torque[finger_dofs]/kp
    layout=e.action.buffers.layout;start=layout.hand_action_starts[layout.sides.index(side)];scale=e.action.buffers.scale.numpy()
    body_names=list(e.model.body_label);obj=body_names.index(f'track3_{name}/object');wrist=body_names.index(f'{side}_sharpa_wave/{side}_hand_C_MC')
    trials=[]
    for limit in json.loads(os.environ.get('V2D_ACTUATION_LIMITS','[0,0.08,0.15,0.30]')):
        e.reset(frame_id=SOURCE_FRAME);e.set_voc_scale(0.)
        initial=e.state_0.body_q.numpy().copy();states=[];joint_errors=[];near_counts=[]
        bias=np.clip(desired,-limit,limit)
        for step in range(60):
            e.timestep.fill_(SOURCE_FRAME)
            lift=.03*float(np.clip((step-19)/20.,0.,1.))
            action=np.zeros(e.action_input.size,np.float32)
            action[start+2]=lift/scale[start+2]
            action[start+6:start+6+len(bias)]=bias/scale[start+6:start+6+len(bias)]
            e.step(wp.array(action,device=e.device));assert np.all(e.object_control_scale.numpy()==0.)
            states.append(e.state_0.body_q.numpy().copy())
            e.solver._update_mjc_data(d,e.model,e.state_0);mujoco.mj_forward(m,d)
            joint_errors.append((d.qpos[finger_q]-np.asarray(row[f'robot_{side}_finger_joints'])[SOURCE_FRAME]).tolist())
            near_counts.append(sum(1 for c in d.contact if c.dist<=.001 and object_id in m.geom_bodyid[c.geom]))
        states=np.asarray(states);actual=states[:,obj];lift_actual=states[-1,wrist,:3]-states[19,wrist,:3]
        drift=float(np.linalg.norm(actual[19,:3]-initial[obj,:3]));follow=float(np.linalg.norm(actual[-1,:3]-(actual[19,:3]+lift_actual)))
        angle=float((Rotation.from_quat(initial[obj,3:]).inv()*Rotation.from_quat(actual[-1,3:])).magnitude())
        trials.append(dict(finger_limit_rad=limit,bias_rad=bias.tolist(),static_drift_m=drift,
            following_error_m=follow,orientation_change_rad=angle,hand_lift_m=float(lift_actual[2]),
            object_lift_m=float(actual[-1,2]-actual[19,2]),near_contact_counts=near_counts,
            final_finger_errors_rad=joint_errors[-1],
            stable_hold_and_lift=bool(drift<=.015 and follow<=.02 and angle<=.3 and lift_actual[2]>=.02)))
    report=dict(diagnostic_only=True,policy_inference=False,source_frame=SOURCE_FRAME,
        object_force_assistance_N=0.,object_torque_assistance_Nm=0.,object_resets_during_trial=False,
        frozen_inputs_modified=False,finger_kp=kp.tolist(),gravity_solution_exists=solution is not None,
        contacts=contacts,gravity_forces_N=solution['forces_N'].tolist() if solution else None,
        requested_finger_preload_rad=desired.tolist(),finger_names=finger_names,trials=trials)
    OUTPUT.parent.mkdir(parents=True,exist_ok=True);OUTPUT.write_text(json.dumps(report,indent=2));print(json.dumps(report),flush=True)
    raise SystemExit(0)


runner.run_training=probe
sys.argv=[sys.argv[0],'experiment=sharpa_flash_sac',
    f"task.parquet='/episode/human_motion_data/{CFG['training_reference']}/processed/sequence_id={CFG['sequence']}/robot_name=sharpa_wave'",
    'task.motion_speed=1.0','task.source_frame_playback=true','scene.world_count=1','reset.reset_finger_openness=0.0',
    'action.input_mode=raw','action.finger_joint_pos_scale=0.15','action.finger_joint_pos_clip=2.0',
    'action.wrist_pos_clip=0.04',f"sim.substeps={CFG.get('physics_substeps',20)}",'sim.nconmax=2048','sim.njmax=4096',
    'logging.mode=disabled','output_dir=/tmp/actuation-probe/checkpoints','hydra.run.dir=/tmp/actuation-probe/hydra']
native=runpy.run_path('/workspace/scripts/train_flash_sac.py',run_name='native_training')


@hydra.main(version_base='1.3',config_path='/workspace/src/flash_chord/configs',config_name='train_flash_sac')
def main(config):
    config=OmegaConf.create(apply_evaluation_composition(OmegaConf.to_container(config,resolve=True),world_count=1,reset_mode='explicit'))
    from flash_chord.embodiments.sharpa_hands import SharpaHandsConfig
    overrides=dict(SharpaHandsConfig().joint_overrides)
    overrides[r'wrist_[xyz]$']={**overrides[r'wrist_[xyz]$'],**CFG['wrist_motor_profile']}
    if os.environ.get('V2D_FINGER_KP'):
        gain=float(os.environ['V2D_FINGER_KP']);assert 1.7<=gain<=40.
        overrides[r'(?:left|right)_(?:thumb|index|middle|ring|pinky)_']={'kp':gain,'kd':.2}
    with open_dict(config.embodiment.config):config.embodiment.config.joint_overrides=overrides
    native['main'].__wrapped__(config)


if __name__=='__main__':main()
