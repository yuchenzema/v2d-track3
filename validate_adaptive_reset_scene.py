"""Exercise the training-only adaptive reset in a captured native physics scene."""
import json,runpy,sys
from pathlib import Path
import hydra,numpy as np,warp as wp
from omegaconf import OmegaConf
import flash_chord.training.flash_sac.runner as runner
from training_finger_reset import install
R=Path('/episode');cfg=json.loads((R/'episode-config.json').read_text())
def probe(environment,*args,**kwargs):
 e=environment.env;reset=e.reset_policy
 assert reset.adaptive_enabled and e._step_graph is not None
 assert e.config.voc.max_force==0. and e.config.voc.max_torque==0.
 assert np.all(e.object_control_scale.numpy()==0.)
 reset.opening_counts.zero_()
 actions=wp.zeros(e.action_input.shape,dtype=wp.float32,device=e.device)
 for _ in range(120):e.step(actions)
 counts=reset.opening_counts.numpy().tolist()
 assert counts[0]>0,('Captured graph did not count completed boundary episodes',counts)
 reset.minimum_frame.assign(np.array([128],np.int32));e.reset()
 starts=e.episode_start_frame.numpy()
 assert np.all((starts>=128)|(starts==124)|(starts==0))
 assert np.any(e.episode_start_frame.numpy()==128)
 assert np.all(e.object_control_scale.numpy()==0.)
 report={'captured_native_scene_contract_passed':True,'world_count':e.world_count,
  'reference_frames':e.reference.num_frames,'exact_boundary_completed_episodes':counts[0],
  'successful_boundary_episodes':counts[1],'boundary_buffer_change_observed_by_reset':True,
  'object_support_force_limit_N':0.,'object_support_torque_limit_Nm':0.,
  'diagnostic_only':True,'policy_improvement_verified':False}
 (R/'research/adaptive-reset-native-scene-contract.json').write_text(json.dumps(report,indent=2))
 print(json.dumps(report),flush=True)
 raise SystemExit(0)
runner.run_training=probe
sys.argv=[sys.argv[0],'experiment=sharpa_flash_sac',
 f"task.parquet='/episode/human_motion_data/{cfg['training_reference']}/processed/sequence_id={cfg['sequence']}/robot_name=sharpa_wave'",
 'task.motion_speed=0.5','task.motion_start_frame=0','task.motion_end_frame=166',
 'scene.world_count=8','sim.substeps=10','sim.njmax=1536','reset.seed=44',
 'reset.reset_finger_openness=1.0','reset.voc_decay_steps=0','env.voc_scale=0.0',
 'env.voc.max_force=0.0','env.voc.max_torque=0.0','logging.mode=disabled',
 'output_dir=/episode/diagnostic-checkpoints','hydra.run.dir=/episode/hydra/adaptive-reset-scene-contract']
native=runpy.run_path('/workspace/scripts/train_flash_sac.py',run_name='native_training')
install(native['main'].__wrapped__.__globals__,adaptive={'minimum_frame':130,'opening_gate_end':166,'stride':2})
@hydra.main(version_base='1.3',config_path='/workspace/src/flash_chord/configs',config_name='train_flash_sac')
def main(config):native['main'].__wrapped__(config)
if __name__=='__main__':main()
