"""Compare free physical contact integration cost; no policy training or assistance."""
import json,os,runpy,sys,time
from pathlib import Path
import hydra,numpy as np,warp as wp
from omegaconf import OmegaConf
import flash_chord.training.flash_sac.runner as runner
from flash_chord.evaluation.harness import apply_evaluation_composition
R=Path('/episode');cfg=json.loads((R/'episode-config.json').read_text());reference=os.environ.get('V2D_BENCHMARK_REFERENCE',cfg['training_reference']);label=os.environ.get('V2D_BENCHMARK_LABEL','contact-scene-benchmark');worlds=int(os.environ.get('V2D_BENCHMARK_WORLDS','128'));assert worlds in [128,512,1024]
def inspect(environment,*args,**kwargs):
 e=environment.env;assert not e.config.auto_reset;e.set_voc_scale(0.)
 frame=round(100*(e.reference.num_frames-1)/(cfg['frames']-1));e.reset(frame_id=frame);action=wp.zeros_like(e.action_input);times=[]
 for i in range(26):
  e.timestep.fill_(frame);wp.synchronize();start=time.perf_counter();e.step(action);wp.synchronize();duration=time.perf_counter()-start
  if i>=6:times.append(duration)
 assert np.isfinite(e.state_0.body_q.numpy()).all() and np.all(e.object_control_scale.numpy()==0.)
 bodies=e.model.body_count//worlds
 np.savez_compressed(R/f'evaluation/{label}-body-properties.npz',body_mass=e.model.body_mass.numpy()[:bodies],body_inertia=e.model.body_inertia.numpy()[:bodies],body_names=list(e.model.body_label)[:bodies])
 report={'reference':reference,'world_count':worlds,'source_frame':100,'warmup_control_steps':6,'measured_control_steps':len(times),
 'mean_control_step_wall_seconds':float(np.mean(times)),'median_control_step_wall_seconds':float(np.median(times)),
 'simulated_environment_steps_per_second':float(worlds/np.mean(times)),'physics_integration_hz':400,
 'object_assistance_zero':True,'physical_states_all_finite':True,'policy_inference':False,'performance_subject_to_other_active_gpu_jobs':True}
 (R/f'evaluation/{label}.json').write_text(json.dumps(report,indent=2));print(report,flush=True);raise SystemExit(0)
runner.run_training=inspect
sys.argv=[sys.argv[0],'experiment=sharpa_flash_sac',f"task.parquet='/episode/human_motion_data/{reference}/processed/sequence_id={cfg['sequence']}/robot_name=sharpa_wave'",'task.motion_speed=0.5','sim.substeps=20',f'scene.world_count={worlds}','reset.reset_finger_openness=0.0','logging.mode=disabled','output_dir=/episode/diagnostic-checkpoints','hydra.run.dir=/episode/hydra/contact-scene-benchmark']
native=runpy.run_path('/workspace/scripts/train_flash_sac.py',run_name='native_training')
@hydra.main(version_base='1.3',config_path='/workspace/src/flash_chord/configs',config_name='train_flash_sac')
def main(config):
 config.sim.nconmax=2048;config.sim.njmax=4096
 config.scene.decompose_objects=not cfg.get('predecomposed_object_collisions',False)
 config=OmegaConf.create(apply_evaluation_composition(OmegaConf.to_container(config,resolve=True),world_count=worlds,reset_mode='explicit'));native['main'].__wrapped__(config)
main()
