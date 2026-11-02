"""Record fixed world0 from the same full native cohort that produces the score.

No success-based world selection, resets, reference retiming or object assistance.
"""
import json,os,runpy
from pathlib import Path
import numpy as np
import jax
import flash_chord.evaluation.harness as harness

if os.environ.get('V2D_ROBOT_TRACKING_FEEDBACK') or os.environ.get('V2D_ROBOT_TRACKING_PROFILES'):
 from robot_object_tracking_feedback import install
 install()

if os.environ.get('V2D_WRIST_BIAS_PROFILES'):
 from diagnostic_wrist_bias_grid import install
 install()

if os.environ.get('V2D_TIMED_WRIST_IMPEDANCE_PROFILE'):
 from timed_wrist_impedance_controller import install
 install()
if os.environ.get('V2D_ACTOR_COMBINATION'):
 from bimanual_policy_controller import install
 install()
if os.environ.get('V2D_RESIDUAL_PROFILE'):
 from residual_gain_controller import install
 install()
if os.environ.get('V2D_GRIP_BIAS') or os.environ.get('V2D_GRIP_BIAS_PROFILES'):
 from grip_bias_controller import install
 from flash_chord.envs.base import BaseEnv
 if not getattr(BaseEnv,'_declared_grip_bias_installed',False):install()

original=harness.record_object_pose_rollout
def record(env,jax_env,*args,**kwargs):
 assert kwargs.get('start_frame',0)==0 and not env.config.auto_reset
 assert env.world_count==int(os.environ['V2D_RECORD_COHORT_SIZE'])
 assert env.world_count>=2 and np.all(env.object_control_scale.numpy()==0.)
 count=env.model.body_count//env.world_count;assert count*env.world_count==env.model.body_count
 poses=[];assistance=[]
 class RecordingSteps:
  def step(self,action):
   result=jax_env.step(action);jax.block_until_ready(result.observation)
   poses.append(env.state_0.body_q.numpy().reshape(env.world_count,count,7)[0].copy())
   assistance.append(float(env.voc_scale.numpy()[0]));return result
 rollout=original(env,RecordingSteps(),*args,**kwargs)
 assert len(poses)==kwargs['step_count'] and np.all(np.asarray(assistance)==0.)
 object_ids=env.command.body_ids_w.numpy().reshape(env.world_count,-1)[0]%count
 np.testing.assert_array_equal(np.asarray(poses)[:,object_ids],rollout.achieved_pose_w[:,0])
 output=Path(os.environ['V2D_STATE_OUTPUT'])
 np.savez_compressed(output,body_poses=poses,body_names=list(env.model.body_label[:count]),reference_frames=np.arange(len(poses)),
  object_assistance=assistance,recording_protocol='Native evaluation cohort, predetermined world0; no success-based selection.',
  simulation_world_count=env.world_count,recorded_world_index=0)
 # Keep all object trajectories so the complete cohort remains auditable.
 np.savez_compressed(output.with_name(output.stem+'-cohort-objects.npz'),achieved=rollout.achieved_pose_w,
  reference=rollout.reference_pose_w,tracking_error=rollout.tracking_error,tracking_error_names=rollout.tracking_error_names,
  recording_protocol='All native cohort object trajectories; recorded video world is fixed0.')
 print('Recorded predetermined world0 and retained complete native cohort object trajectories.',flush=True)
 return rollout
harness.record_object_pose_rollout=record
runpy.run_path('/episode/scripts/evaluate_reference_controller.py',run_name='__main__')
