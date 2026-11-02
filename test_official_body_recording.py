import os
from pathlib import Path
import sys
import tempfile
import types
import unittest
from unittest.mock import patch
import numpy as np
import official_body_recording


class Array:
    def __init__(self,value):self.value=np.asarray(value,float)
    def numpy(self):return self.value


class OfficialRecordingTests(unittest.TestCase):
    def test_float32_quaternion_rounding_does_not_create_a_false_rotation(self):
        from official_policy_suite import quaternion_discrepancy
        from scipy.spatial.transform import Rotation
        q=Rotation.from_rotvec([[.4,.2,.1],[.9,.3,-.4]]).as_quat().astype(np.float32)
        self.assertLess(quaternion_discrepancy(q,q.copy()),1e-7)
        displaced=(Rotation.from_quat(q)*Rotation.from_rotvec([0,0,.002])).as_quat()
        self.assertGreater(quaternion_discrepancy(q,displaced),.0019)
    def test_capture_observes_the_same_steps_and_restores_step_method(self):
        with tempfile.TemporaryDirectory() as directory:
            output=Path(directory)/'states.npz'
            env=types.SimpleNamespace(world_count=1,state_0=types.SimpleNamespace(body_q=Array(np.zeros((2,7)))),
                model=types.SimpleNamespace(body_label=['hand','object']),voc_scale=Array([0.]))
            calls=[]
            def step(action):
                calls.append(action);env.state_0.body_q.value[:,0]+=1
                return types.SimpleNamespace(observation='observation')
            jax_env=types.SimpleNamespace(step=step)
            observed=[]
            def rollout(env,jax_env,*args,step_count,**kwargs):
                for index in range(step_count):
                    jax_env.step(index);observed.append(env.state_0.body_q.numpy().copy())
                return 'unchanged rollout result'
            package=types.ModuleType('flash_chord');evaluation=types.ModuleType('flash_chord.evaluation')
            harness=types.ModuleType('flash_chord.evaluation.harness');harness.record_object_pose_rollout=rollout
            package.evaluation=evaluation;evaluation.harness=harness
            modules={'flash_chord':package,'flash_chord.evaluation':evaluation,'flash_chord.evaluation.harness':harness}
            with patch.dict(sys.modules,modules),patch.dict(os.environ,{'V2D_OFFICIAL_BODY_STATES':str(output),'V2D_OFFICIAL_CHECKPOINT_SHA256':'actual-checkpoint'}):
                official_body_recording.install()
                result=harness.record_object_pose_rollout(env,jax_env,step_count=3,start_frame=0)
            self.assertEqual(result,'unchanged rollout result')
            self.assertEqual(calls,[0,1,2]);self.assertIs(jax_env.step,step)
            with np.load(output,allow_pickle=False) as record:
                np.testing.assert_array_equal(record['body_poses'],np.asarray(observed))
                np.testing.assert_array_equal(record['reference_frames'],[0,1,2])
                self.assertTrue(record['policy_inference'])
                self.assertEqual(str(record['checkpoint_sha256']),'actual-checkpoint')


if __name__=='__main__':unittest.main()
