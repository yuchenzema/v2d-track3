import unittest
import cv2
import numpy as np
from sift_metric_odometry import estimate_relative


class MetricCameraTests(unittest.TestCase):
    def test_metric_translation_recovered_with_outliers(self):
        rng=np.random.default_rng(21);xyz=rng.uniform([-.5,-.3,.6],[.5,.3,2.],(100,3))
        K=np.array([[600.,0,512],[0,600,384],[0,0,1.]])
        rv=np.array([.01,-.02,.005]);tv=np.array([.018,-.008,.009])
        uv=cv2.projectPoints(xyz,rv,tv,K,None)[0][:,0];uv[:20]=rng.uniform([0,0],[1024,768],(20,2))
        M,report=estimate_relative(xyz,uv,K)
        np.testing.assert_allclose(M[:3,3],tv,atol=1e-5)
        self.assertGreaterEqual(report['inliers'],75)
        # World-from-current must compose the inverse of current-from-previous.
        world=np.linalg.inv(M);np.testing.assert_allclose(world@M,np.eye(4),atol=1e-9)

    def test_insufficient_background_stops_instead_of_freezing_camera(self):
        with self.assertRaises(RuntimeError):estimate_relative(np.ones((5,3)),np.ones((5,2)),np.eye(3))
