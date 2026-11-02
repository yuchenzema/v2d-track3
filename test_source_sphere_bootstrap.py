import unittest
import cv2
import numpy as np
from scipy.spatial.transform import Rotation
from source_sphere_bootstrap import detect,pose_candidates


class SphereBootstrapTests(unittest.TestCase):
    def test_caps_connected_to_narrow_stems_and_hand_occlusion(self):
        image=np.full((100,160,3),(30,180,60),np.uint8)
        for x in [45,110]:
            cv2.circle(image,(x,40),6,(150,150,150),-1)
            cv2.line(image,(x,45),(x,59),(150,150,150),1)
        mask=np.ones((100,160),np.uint8);hand=np.zeros_like(mask);depth=np.ones(mask.shape)
        intr=dict(fx=500,fy=500,cx=80,cy=50)
        template=dict(markers=[dict(radius_m=.012),dict(radius_m=.012)])
        observations=detect(image,mask,hand,depth,intr,template)
        self.assertEqual(len(observations),2)
        np.testing.assert_allclose(sorted([o['xy'] for o in observations]),[[45,40],[110,40]],atol=.5)
        hand[30:65,100:125]=1
        self.assertEqual(len(detect(image,mask,hand,depth,intr,template)),1)
        depth[:]=np.nan
        self.assertEqual(detect(image,mask,hand,depth,intr,template),[])

    def test_metric_pair_recovers_proper_pose_without_initial_pose(self):
        markers=[dict(center_m=[-.02,0,0],radius_m=.007),dict(center_m=[.02,0,0],radius_m=.007)]
        axis=np.array([.3,.4,.5]);axis/=np.linalg.norm(axis)
        initial=Rotation.align_vectors(axis[None],np.array([[1.,0,0]]))[0].as_matrix()
        expected=Rotation.from_rotvec(axis*np.pi/3).as_matrix()@initial;translation=np.array([.1,-.2,.7]);observations=[]
        for marker in markers:
            centre=expected@marker['center_m']+translation;ray=centre/np.linalg.norm(centre)
            observations.append(dict(surface_xyz_m=(centre-marker['radius_m']*ray).tolist(),unit_ray=ray.tolist()))
        candidates=pose_candidates(dict(markers=markers),observations,roll_samples=12)
        self.assertTrue(candidates)
        error=[Rotation.from_matrix(c['pose'][:3,:3]@expected.T).magnitude() for c in candidates]
        best=candidates[int(np.argmin(error))]['pose']
        self.assertLess(min(error),1e-8);np.testing.assert_allclose(best[:3,3],translation,atol=1e-10)
        for item in candidates:self.assertAlmostEqual(np.linalg.det(item['pose'][:3,:3]),1.,places=8)
        observations[1]['surface_xyz_m']=[1.,1.,1.]
        self.assertEqual(pose_candidates(dict(markers=markers),observations),[])


if __name__=='__main__':unittest.main()
