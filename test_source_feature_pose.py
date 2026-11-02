import unittest

import cv2
import numpy as np
from scipy.spatial.transform import Rotation

from source_feature_pose import track,penalty


class SourceFeaturePoseTests(unittest.TestCase):
    def scene(self):
        intr=dict(fx=700.,fy=700.,cx=512.,cy=384.)
        rng=np.random.default_rng(7)
        image=np.zeros((768,1024,3),np.uint8)
        image[220:550,300:700]=rng.integers(30,225,(330,400,3),dtype=np.uint8)
        image=cv2.GaussianBlur(image,(3,3),0)
        mask=np.zeros(image.shape[:2],np.uint8);mask[220:550,300:700]=1
        warp=np.array([[1.,0.,7.],[0.,1.,-4.]])
        moved=cv2.warpAffine(image,warp,(1024,768))
        moved_mask=cv2.warpAffine(mask,warp,(1024,768),flags=cv2.INTER_NEAREST)>0
        surface=cv2.resize(mask,(512,384),interpolation=cv2.INTER_NEAREST)*.6
        class Renderer:
            def depth(self,pose):return surface
        initial=np.eye(4);initial[2,3]=.6
        return image,moved,mask.astype(bool),moved_mask,Renderer(),initial,intr

    def test_measured_pixel_motion_recovers_metric_rigid_translation(self):
        a,b,ma,mb,renderer,pose,intr=self.scene()
        proposal=track(a,b,renderer,pose,intr,ma,mb)
        self.assertIsNotNone(proposal)
        expected=pose[:3,3]+np.array([7.*.6/700.,-4.*.6/700.,0.])
        np.testing.assert_allclose(proposal['pose'][:3,3],expected,atol=.002)
        self.assertLess(Rotation.from_matrix(proposal['pose'][:3,:3]).magnitude(),.02)
        self.assertGreaterEqual(proposal['inlier_count'],8)
        self.assertLess(proposal['median_forward_backward_error_px'],.3)

    def test_occluded_source_pixels_cannot_generate_a_pose(self):
        a,b,ma,mb,renderer,pose,intr=self.scene()
        self.assertIsNone(track(a,b,renderer,pose,intr,ma,mb,np.ones_like(ma)))
        self.assertIsNone(track(a,b,renderer,pose,intr,ma,mb,current_occlusion=np.ones_like(mb)))

    def test_unrelated_images_do_not_supply_object_motion(self):
        a,_,ma,mb,renderer,pose,intr=self.scene()
        b=np.random.default_rng(19).integers(0,256,a.shape,dtype=np.uint8)
        self.assertIsNone(track(a,b,renderer,pose,intr,ma,mb))

    def test_verified_current_markers_can_override_inherited_wrong_correspondences(self):
        a,b,ma,mb,renderer,pose,intr=self.scene()
        proposal=track(a,b,renderer,pose,intr,ma,mb)
        good=proposal['pose'];wrong=good.copy()
        wrong[:3,:3]=Rotation.from_rotvec([np.pi,0.,0.]).as_matrix()
        self.assertLess(penalty(good,proposal,intr),.01)
        self.assertGreater(penalty(wrong,proposal,intr),1.)
        self.assertEqual(penalty(wrong,proposal,intr,independent_markers=True),0.)
