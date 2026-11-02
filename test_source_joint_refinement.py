import unittest
import tempfile
from pathlib import Path
import cv2
import numpy as np
from scipy.spatial.transform import Rotation
from source_photometric_reference import MeshPhotometricReference
from source_static_joint_fit import stable_windows
from source_static_joint_fit import refine
from source_mask_rasterizer import MeshSilhouette
from source_cad_alignment import human_mask,object_visibility,metrics,qualifies


class PhotometricTests(unittest.TestCase):
    def fixture(self):
        u,v=np.meshgrid(np.arange(215,296,10),np.arange(150,231,10));u=u.ravel();v=v.ravel()
        points=np.c_[(u-256)/200,(v-192)/200,np.ones(len(u))]
        colors=np.random.default_rng(319).uniform(.1,.9,(len(u),3));image=np.full((384,512,3),.9,np.float32)
        for x,y,color in zip(u,v,colors):image[y-1:y+2,x-1:x+2]=color
        ref=MeshPhotometricReference(points,colors);intr=dict(fx=400,fy=400,cx=512,cy=384)
        return ref,image,intr,np.ones((384,512)),np.ones((384,512),bool)

    def test_texture_alignment_rejects_shifted_pose(self):
        ref,image,intr,depth,valid=self.fixture();pose=np.eye(4)
        good,count=ref.residual(pose,image,intr,depth,valid)
        pose[0,3]=.03;bad,_=ref.residual(pose,image,intr,depth,valid)
        self.assertGreater(count,30);self.assertLess(good,1e-6);self.assertGreater(bad,good+.05)

    def test_exposure_change_is_normalized(self):
        ref,image,intr,depth,valid=self.fixture()
        loss,_=ref.residual(np.eye(4),image*.6+.1,intr,depth,valid)
        self.assertLess(loss,1e-6)

    def test_occluded_samples_do_not_create_photometric_evidence(self):
        ref,image,intr,depth,valid=self.fixture()
        loss,count=ref.residual(np.eye(4),image,intr,depth*2,valid)
        self.assertEqual(count,0);self.assertEqual(loss,0.)


class StaticWindowTests(unittest.TestCase):
    def test_joint_fit_rechecks_all_frames_before_freezing(self):
        with tempfile.TemporaryDirectory() as directory:
            root=Path(directory);rec=root/'reconstruction'
            for folder in ['frames','masks/1','depth_stereo']:(rec/folder).mkdir(parents=True)
            intr=dict(fx=400,fy=400,cx=512,cy=384)
            renderer=MeshSilhouette([[-.2,-.2,0.],[.2,-.2,0.],[0.,.2,0.]],[[0,1,2]],intr)
            world=np.eye(4);world[2,3]=1.;mask=cv2.resize(renderer.render(world).astype(np.uint8)*255,(1024,768),interpolation=cv2.INTER_NEAREST)
            for frame in range(10):
                cv2.imwrite(str(rec/f'frames/{frame:06d}.png'),np.full((768,1024,3),128,np.uint8))
                cv2.imwrite(str(rec/f'masks/1/{frame:06d}.png'),mask)
                cv2.imwrite(str(rec/f'depth_stereo/{frame:06d}.png'),np.full((768,1024),32767,np.uint16))
            objects=np.repeat(world[None,None],10,axis=0);objects[:,0,0,3]=np.sin(np.arange(10))*.0001
            motion=dict(camera_to_world=np.repeat(np.eye(4)[None],10,axis=0),object_to_world=objects)
            records=[dict(frame=i,object='fixture',visibility='visible',sphere_observations=[]) for i in range(10)]
            result=refine(root,dict(frames=10,objects=['fixture']),motion,{'fixture':renderer},{},records,{'observed':np.zeros((10,2),bool)},intr,human_mask,object_visibility,metrics,qualifies,{})
            self.assertEqual(result['accepted_windows'],1)
            np.testing.assert_allclose(motion['object_to_world'],np.broadcast_to(motion['object_to_world'][0],motion['object_to_world'].shape))
            self.assertTrue(all(row['static_joint_fit'] for row in records))
    def test_stationary_jitter_is_a_candidate(self):
        poses=np.repeat(np.eye(4)[None],20,axis=0);poses[:,0,3]=np.sin(np.arange(20))*.0004
        self.assertEqual(stable_windows(poses,[True]*20),[(0,20)])

    def test_translating_or_rotating_object_is_not_frozen(self):
        poses=np.repeat(np.eye(4)[None],20,axis=0);poses[:,0,3]=np.arange(20)*.01
        self.assertEqual(stable_windows(poses,[True]*20),[])
        poses[:,0,3]=0;poses[:,:3,:3]=Rotation.from_euler('z',(np.arange(20)*.2)[:,None]).as_matrix()
        self.assertEqual(stable_windows(poses,[True]*20),[])

    def test_hand_contact_splits_static_hypotheses(self):
        poses=np.repeat(np.eye(4)[None],20,axis=0);eligible=[True]*20;eligible[10]=False
        self.assertEqual(stable_windows(poses,eligible),[(0,10),(11,20)])


if __name__=='__main__':unittest.main()
