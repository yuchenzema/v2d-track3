"""Per-grasp unassisted reference playback from before approach; no learned actor."""
import json,runpy,sys,hashlib
from pathlib import Path
import numpy as np
import warp as wp
import mujoco
import hydra
from omegaconf import OmegaConf
import flash_chord.training.flash_sac.runner as runner
from flash_chord.evaluation.harness import apply_evaluation_composition
from grasp_physics_checks import gravity_cone_feasible,native_body_id
from legacy_source_frame_playback import install as install_source_frame_playback
install_source_frame_playback()

ROOT=Path('/episode');cfg=json.loads((ROOT/'episode-config.json').read_text());reference=cfg['training_reference']


def probe(env,*args,**kwargs):
    e=env.env;assert not e.config.auto_reset and e.reference.num_frames==cfg['frames']
    assert e.config.voc.max_force==0. and e.config.voc.max_torque==0.
    e.set_voc_scale(0.);action=wp.zeros_like(e.action_input);names=list(e.model.body_label)
    target=np.asarray(e.reference.object_body_pos_w());rows=[]
    m=e.solver.mj_model;data=mujoco.MjData(m)
    for index,g in enumerate(cfg['grasps']):
        body=g['object'];name=cfg['objects'][body];start=g['start'];end=g['end'];lo=max(0,start-12)
        displacement=np.linalg.norm(target[start:end+1,body]-target[start,body],axis=1)
        moved=np.flatnonzero(displacement>.015);hi=min(end,max(start+20,start+int(moved[0])+8 if len(moved) else start+20))
        e.reset(frame_id=lo);e.set_voc_scale(0.);body_id=names.index(f'track3_{name}/object')
        initial=e.state_0.body_q.numpy()[body_id,:3].copy();errors=[];achieved=[];support=[]
        for frame in range(lo,hi+1):
            assert int(e.timestep.numpy()[0])==frame
            e.step(action);assert np.all(e.object_control_scale.numpy()==0.)
            position=e.state_0.body_q.numpy()[body_id,:3];achieved.append(position.copy())
            if frame>=start:errors.append(float(np.linalg.norm(position-target[frame,body])))
            e.solver._update_mjc_data(data,e.model,e.state_0);mujoco.mj_forward(m,data)
            native_id=native_body_id(lambda label:mujoco.mj_name2id(m,mujoco.mjtObj.mjOBJ_BODY,label),f'track3_{name}/object')
            normal=[];points=[];friction=[]
            for contact in data.contact:
                ids=[int(m.geom_bodyid[int(geom)]) for geom in contact.geom]
                if native_id not in ids or contact.dist>.001:continue
                other=ids[1] if ids[0]==native_id else ids[0]
                other_name=mujoco.mj_id2name(m,mujoco.mjtObj.mjOBJ_BODY,other) or ''
                if 'sharpa_wave' not in other_name:continue
                normal.append(np.asarray(contact.frame[:3])*(1. if ids[1]==native_id else -1.));points.append(contact.pos.copy());friction.append(float(contact.friction[0]))
            if frame>=start and position[2]>initial[2]+.01:
                support.append(gravity_cone_feasible(normal,points,friction,float(m.body_mass[native_id]),data.xipos[native_id],gravity=m.opt.gravity))
        actual=np.asarray(achieved);expected=target[lo:hi+1,body];expected_motion=float(np.max(np.linalg.norm(expected-expected[0],axis=1)));actual_motion=float(np.max(np.linalg.norm(actual-actual[0],axis=1)))
        passed=bool(errors and max(errors)<=.02 and (expected_motion<.015 or actual_motion>=.75*expected_motion) and (not support or np.mean(support)>=.8))
        rows.append(dict(grasp=index,object=name,side=g['side'],source_start=lo,source_end=hi,
                         observed_pre_approach_frames=start-lo,initially_held=start==0,
                         maximum_object_position_error_m=max(errors) if errors else None,expected_motion_m=expected_motion,actual_motion_m=actual_motion,
                         lifted_contact_frames_tested=len(support),gravity_cone_feasible_fraction=float(np.mean(support)) if support else None,
                         passed=passed))
    report=dict(diagnostic_only=True,policy_inference=False,source_frame_playback=True,object_force_assistance_N=0.,object_torque_assistance_Nm=0.,
                reference_parquet_sha256=hashlib.sha256((ROOT/f"human_motion_data/{reference}/processed/sequence_id={cfg['sequence']}/robot_name=sharpa_wave/motion.parquet").read_bytes()).hexdigest(),
                resets_during_each_probe=False,starts_before_approach_when_observed=True,grasps=rows,all_grasps_passed=bool(rows) and all(row['passed'] for row in rows),
                mass_and_friction_are_simulation_parameters_not_measured_human_object_properties=True,normal_force_limit_per_contact_N=25.)
    (ROOT/'research/approach-grasp-probes.json').write_text(json.dumps(report,indent=2));raise SystemExit(0)


runner.run_training=probe
sys.argv=[sys.argv[0],'experiment=sharpa_flash_sac',f"task.parquet='/episode/human_motion_data/{reference}/processed/sequence_id={cfg['sequence']}/robot_name=sharpa_wave'",
          'task.motion_speed=1.0','task.source_frame_playback=true',f"sim.substeps={cfg.get('physics_substeps',20)}",'sim.nconmax=2048','sim.njmax=4096',
          'reset.reset_finger_openness=0.0','scene.world_count=1','logging.mode=disabled','output_dir=/episode/diagnostic-checkpoints','hydra.run.dir=/episode/hydra/approach-grasp-probe']
native=runpy.run_path('/workspace/scripts/train_flash_sac.py',run_name='native_training')


@hydra.main(version_base='1.3',config_path='/workspace/src/flash_chord/configs',config_name='train_flash_sac')
def main(config):
    config=OmegaConf.create(apply_evaluation_composition(OmegaConf.to_container(config,resolve=True),world_count=1,reset_mode='explicit'))
    from flash_chord.embodiments.sharpa_hands import SharpaHandsConfig
    from omegaconf import open_dict
    overrides=dict(SharpaHandsConfig().joint_overrides)
    overrides[r'wrist_[xyz]$']={**overrides[r'wrist_[xyz]$'],**cfg['wrist_motor_profile']}
    with open_dict(config.embodiment.config):config.embodiment.config.joint_overrides=overrides
    native['main'].__wrapped__(config)


main()
