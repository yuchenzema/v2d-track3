import unittest
import numpy as np
from source_mask_rasterizer import MeshSilhouette
from metric_surface_localisation import translation_seed


class MetricSurfaceTests(unittest.TestCase):
    def test_metric_surface_recovers_large_depth_initialisation_error(self):
        intr=dict(fx=400,fy=400,cx=512,cy=384)
        renderer=MeshSilhouette([[-.2,-.2,0.],[.2,-.2,0.],[0.,.2,0.]],[[0,1,2]],intr)
        correct=np.eye(4);correct[2,3]=.8;measured=renderer.depth(correct)
        wrong=correct.copy();wrong[2,3]=1.1
        proposal=translation_seed(renderer,wrong,intr,measured,measured>0)
        self.assertIsNotNone(proposal);self.assertAlmostEqual(proposal[2,3],.8,places=5)
        self.assertLess(np.linalg.norm(proposal[:3,3]-correct[:3,3]),.003)

    def test_missing_metric_measurements_do_not_invent_a_pose(self):
        intr=dict(fx=400,fy=400,cx=512,cy=384)
        renderer=MeshSilhouette([[-.2,-.2,0.],[.2,-.2,0.],[0.,.2,0.]],[[0,1,2]],intr)
        pose=np.eye(4);pose[2,3]=.8
        self.assertIsNone(translation_seed(renderer,pose,intr,np.zeros((384,512)),np.ones((384,512),bool)))

    def test_occlusion_does_not_translate_an_already_correct_object(self):
        intr=dict(fx=400,fy=400,cx=512,cy=384)
        renderer=MeshSilhouette([[-.2,-.2,0.],[.2,-.2,0.],[.2,.2,0.],[-.2,.2,0.]],[[0,1,2],[0,2,3]],intr)
        correct=np.eye(4);correct[2,3]=.8;measured=renderer.depth(correct)
        visible=np.ones(measured.shape,bool);visible[:,256:]=False
        target=(measured>0)&visible
        biased=translation_seed(renderer,correct,intr,measured,target)
        self.assertGreater(np.linalg.norm(biased[:3,3]-correct[:3,3]),.05)
        proposal=translation_seed(renderer,correct,intr,measured,target,visibility=visible)
        np.testing.assert_allclose(proposal,correct,atol=1e-12)

    def test_no_observable_model_surface_does_not_generate_a_seed(self):
        intr=dict(fx=400,fy=400,cx=512,cy=384)
        renderer=MeshSilhouette([[-.2,-.2,0.],[.2,-.2,0.],[0.,.2,0.]],[[0,1,2]],intr)
        pose=np.eye(4);pose[2,3]=.8;depth=renderer.depth(pose)
        self.assertIsNone(translation_seed(renderer,pose,intr,depth,depth>0,visibility=np.zeros(depth.shape,bool)))


if __name__=='__main__':unittest.main()
