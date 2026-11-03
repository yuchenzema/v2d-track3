import unittest
import numpy as np
from scipy.spatial.transform import Rotation
from fast_metric_source_pose import rigid_fit


class MetricSourcePoseTests(unittest.TestCase):
    def test_recovers_small_rigid_motion_without_changing_metric_scale(self):
        source=np.random.default_rng(1).normal(size=(384,3))*.1
        expected=np.eye(4);expected[:3,:3]=Rotation.from_rotvec([.02,-.03,.01]).as_matrix();expected[:3,3]=[.1,-.2,.4]
        target=source@expected[:3,:3].T+expected[:3,3]
        actual,error=rigid_fit(source,target,np.eye(4))
        np.testing.assert_allclose(actual,expected,atol=1e-6)
        self.assertLess(error,1e-6);self.assertAlmostEqual(np.linalg.det(actual[:3,:3]),1.)
