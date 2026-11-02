"""CPU integration contract against the actual cup scene and native objective."""
import json,runpy,sys
from pathlib import Path
import numpy as np,warp as wp
import flash_chord.training.flash_sac.runner as runner
from flash_chord.objectives.tracking import TrackingObjective
from translation_precision_reward import install
original=TrackingObjective.evaluate;install(fraction=.25,scale=.01)
R=Path('/episode');cfg=json.loads((R/'episode-config.json').read_text())
def inspect(environment,*args,**kwargs):
 e=environment.env;assert e.world_count==1;obj=e.objective;records=[]
 for offset in [0.,.01,.04]:
  e.reset(frame_id=120);e.command.gather(e.timestep)
  poses=e.state_0.body_q.numpy();ids=e.command.body_ids.numpy();poses[ids,0]+=offset;e.state_0.body_q.assign(poses)
  original(obj,e.state_0,e.timestep,e.terminated);native=obj.buffers.object_keypoints.numpy().copy()
  obj.evaluate(e.state_0,e.timestep,e.terminated)
  distances=np.linalg.norm(poses[ids,:3]-e.command.body_target_pos_w.numpy(),axis=1)
  np.testing.assert_allclose(distances,offset,atol=1e-6)
  expected=.75*native+.25*np.exp(-distances/.01).mean()
  np.testing.assert_allclose(obj.buffers.object_keypoints.numpy(),expected,rtol=2e-5)
  terms=obj.terms.numpy().reshape(1,-1);expected_score=(terms*obj.weights.numpy()).sum(1)*obj.buffers.frame_dt
  np.testing.assert_allclose(obj.score.numpy(),expected_score,rtol=2e-5)
  np.testing.assert_array_equal(e.state_0.body_q.numpy(),poses)
  records.append({'test_offset_m':offset,'native_object_term':native.tolist(),'precision_mixed_term':expected.tolist(),'score_consistent':True})
 report={'actual_scene_cpu_integration_passed':True,'diagnostic_only':True,'no_training_started':True,'object_states_untouched_by_objective':True,'fraction':.25,'scale_m':.01,'tests':records}
 (R/'research/translation-precision-integration-contract.json').write_text(json.dumps(report,indent=2));print(json.dumps(report),flush=True);raise SystemExit(0)
runner.run_training=inspect
sys.argv=[sys.argv[0],'experiment=sharpa_flash_sac',f"task.parquet='/episode/human_motion_data/{cfg['training_reference']}/processed/sequence_id={cfg['sequence']}/robot_name=sharpa_wave'",'task.motion_speed=0.5','scene.world_count=1','training.device=cpu','logging.mode=disabled','reset.reset_finger_openness=0.0','sim.nconmax=2048','sim.njmax=4096','hydra.run.dir=/episode/hydra/precision-reward-contract']
from flash_chord.envs.rl import RLEnv
RLEnv.capture_step=lambda self:None
runpy.run_path('/workspace/scripts/train_flash_sac.py',run_name='__main__')
