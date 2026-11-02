"""Diagnostic only: test whether unheld final video poses remain at rest under gravity."""
import json,os,runpy,sys
from pathlib import Path
import hydra,numpy as np,warp as wp
from scipy.spatial.transform import Rotation
from omegaconf import OmegaConf
import flash_chord.training.flash_sac.runner as runner
from flash_chord.evaluation.harness import apply_evaluation_composition
r=Path('/episode');cfg=json.loads((r/'episode-config.json').read_text());reference=os.environ.get('V2D_DIAGNOSTIC_REFERENCE',cfg.get('training_reference','track3_coherent_variant'));substeps=int(os.environ.get('V2D_DIAGNOSTIC_SUBSTEPS',cfg.get('physics_substeps',5)));label=os.environ.get('V2D_DIAGNOSTIC_LABEL','terminal-support-diagnostic')
def inspect(environment,*args,**kwargs):
 e=environment.env;assert not e.config.auto_reset;e.set_voc_scale(0.);last=e.reference.num_frames-1
 if os.environ.get('V2D_DIAGNOSTIC_SOURCE_FRAME'):
  source=int(os.environ['V2D_DIAGNOSTIC_SOURCE_FRAME']);assert 0<=source<cfg['frames'];last=round(source*(e.reference.num_frames-1)/(cfg['frames']-1))
 e.reset(frame_id=last);initial=e.state_0.body_q.numpy().copy();poses=[];assistance=[];action=wp.zeros_like(e.action_input)
 for i in range(40):
  # Freeze only the reference clock. Physical states are never reset after
  # initialization; the objects evolve continuously without any virtual force.
  e.timestep.fill_(last);e.step(action);poses.append(e.state_0.body_q.numpy().copy());assistance.append(float(e.voc_scale.numpy()[0]))
 assert np.all(np.array(assistance)==0.);poses=np.array(poses);names=list(e.model.body_label);rows=[]
 for b,name in enumerate(cfg['objects']):
  if any(g['object']==b and g['end']>=cfg['frames']-1 for g in cfg['grasps']):continue
  k=names.index(f'track3_{name}/object');drift=np.linalg.norm(poses[:,k,:3]-initial[k,:3],axis=1);angle=(Rotation.from_quat(initial[k,3:]).inv()*Rotation.from_quat(poses[:,k,3:])).magnitude();rows.append({'object':name,'maximum_position_drift_m':float(drift.max()),'final_position_drift_m':float(drift[-1]),'maximum_rotation_drift_rad':float(angle.max()),'final_rotation_drift_rad':float(angle[-1]),'passed_1cm_0_2rad_rest_test':bool(drift.max()<=.01 and angle.max()<=.2)})
 np.savez_compressed(r/f'evaluation/{label}-states.npz',body_poses=poses,initial_body_poses=initial,body_names=names,object_assistance=assistance,trajectory_kind=__doc__);report={'method':__doc__,'policy_inference':False,'object_assistance':0.,'single_continuous_physical_test':True,'object_resets_after_initialization':False,'reference_clock_held_at_final_frame':not bool(os.environ.get('V2D_DIAGNOSTIC_SOURCE_FRAME')),'frozen_source_frame':int(os.environ.get('V2D_DIAGNOSTIC_SOURCE_FRAME',cfg['frames']-1)),'reference':reference,'physical_duration_s':2.,'physics_integration_hz':20*substeps,'all_states_finite':bool(np.isfinite(poses).all()),'objects':rows};(r/f'evaluation/{label}.json').write_text(json.dumps(report,indent=2));print(cfg['episode'],report,flush=True);raise SystemExit(0)
runner.run_training=inspect;sys.argv=[sys.argv[0],'experiment=sharpa_flash_sac',f"task.parquet='/episode/human_motion_data/{reference}/processed/sequence_id={cfg['sequence']}/robot_name=sharpa_wave'",'task.motion_speed=0.5',f'sim.substeps={substeps}','reset.reset_finger_openness=0.0','scene.world_count=1','logging.mode=disabled','output_dir=/episode/diagnostic-checkpoints','hydra.run.dir=/episode/hydra/terminal-support-diagnostic'];native=runpy.run_path('/workspace/scripts/train_flash_sac.py',run_name='native_training')
@hydra.main(version_base='1.3',config_path='/workspace/src/flash_chord/configs',config_name='train_flash_sac')
def main(config):
 config.sim.nconmax=int(cfg.get('physics_nconmax',512));config.sim.njmax=int(cfg.get('physics_njmax',1536))
 config=OmegaConf.create(apply_evaluation_composition(OmegaConf.to_container(config,resolve=True),world_count=1,reset_mode='explicit'));native['main'].__wrapped__(config)
main()
