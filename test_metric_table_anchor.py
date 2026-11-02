import unittest
import numpy as np
from metric_table_anchor import anchored_cameras


class StaticTableTests(unittest.TestCase):
    def test_noisy_circle_outliers_do_not_distort_measured_camera_path(self):
        rng=np.random.default_rng(7);count=40
        slam=np.repeat(np.eye(4)[None],count,axis=0);slam[:,0,3]=np.linspace(0,.12,count)
        true_center=np.array([.02,-.1,.7]);centers=true_center-slam[:,:3,3]+rng.normal(0,.001,(count,3));centers[-8:]+=[.3,.2,0.]
        normals=np.repeat([[0.,0.,-1.]],count,axis=0)
        cameras,report=anchored_cameras(slam,centers,normals,np.ones(count))
        self.assertTrue(set(range(32,40))<=set(report['rejected_table_measurement_frames']))
        for frame in range(1,count):
            np.testing.assert_allclose(np.linalg.inv(cameras[frame-1])@cameras[frame],np.linalg.inv(slam[frame-1])@slam[frame],atol=1e-9)
        self.assertLess(report['measured_center_residual_median_m'],.004)

    def test_no_measured_table_anchor_is_not_silently_invented(self):
        count=10;slam=np.repeat(np.eye(4)[None],count,axis=0)
        with self.assertRaises(RuntimeError):anchored_cameras(slam,np.ones((count,3)),np.ones((count,3)),np.zeros(count))
