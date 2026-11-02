import unittest
import cv2
import numpy as np
import trimesh
from scipy.spatial.transform import Rotation
from source_mask_rasterizer import MeshSilhouette
from contour_pose_refinement import propose,face_flip_proposals,temporal_pose_cost,moving_orientation_proposals,surface_center


class ContourRecoveryTests(unittest.TestCase):
    def unevenly_tessellated_box(self):
        mesh=trimesh.creation.box(extents=[.2,.12,.01]);mesh.apply_translation([.1,-.03,.02])
        for _ in range(3):
            selected=np.flatnonzero(mesh.triangles_center[:,0]>.15)
            vertices,faces=trimesh.remesh.subdivide(mesh.vertices,mesh.faces,face_index=selected)
            mesh=trimesh.Trimesh(vertices=vertices,faces=faces,process=False)
        self.assertGreater(np.linalg.norm(mesh.vertices.mean(0)-mesh.centroid),.02)
        return mesh,MeshSilhouette(mesh.vertices,mesh.faces,dict(fx=700.,fy=700.,cx=512.,cy=384.))

    def test_moving_rotation_preserves_surface_centre_despite_uneven_tessellation(self):
        mesh,renderer=self.unevenly_tessellated_box()
        np.testing.assert_allclose(surface_center(renderer),[.1,-.03,.02],atol=1e-12)
        seed=np.eye(4);seed[:3,3]=[.15,-.06,.7]
        previous=np.eye(4);previous[:3,:3]=Rotation.from_rotvec([.1,-.2,.3]).as_matrix()
        for pose in moving_orientation_proposals(renderer,seed,np.eye(4),previous):
            np.testing.assert_allclose(pose[:3,:3]@mesh.centroid+pose[:3,3],seed[:3,:3]@mesh.centroid+seed[:3,3],atol=1e-12)

    def test_face_flips_preserve_surface_centre_despite_uneven_tessellation(self):
        mesh,renderer=self.unevenly_tessellated_box()
        seed=np.eye(4);seed[:3,:3]=Rotation.from_rotvec([.2,-.1,.3]).as_matrix();seed[:3,3]=[.03,.02,.7]
        for pose in face_flip_proposals(renderer,seed):
            np.testing.assert_allclose(pose[:3,:3]@mesh.centroid+pose[:3,3],seed[:3,:3]@mesh.centroid+seed[:3,3],atol=1e-12)

    def test_moving_thin_object_keeps_current_centroid_and_continues_real_rotation(self):
        mesh=trimesh.creation.box(extents=[.2,.12,.01]);mesh.apply_translation([.1,-.03,.02])
        renderer=MeshSilhouette(mesh.vertices,mesh.faces,dict(fx=700.,fy=700.,cx=512.,cy=384.))
        camera=np.eye(4);camera[:3,:3]=Rotation.from_rotvec([.1,.2,-.3]).as_matrix()
        earlier=np.eye(4);previous=np.eye(4);previous[:3,:3]=Rotation.from_rotvec([0,0,.1]).as_matrix()
        seed=np.eye(4);seed[:3,3]=[.15,-.06,.7];before=seed.copy()
        poses=moving_orientation_proposals(renderer,seed,camera,previous,earlier);center=mesh.vertices.mean(0)
        for pose in poses:
            np.testing.assert_allclose(pose[:3,:3]@center+pose[:3,3],seed[:3,:3]@center+seed[:3,3],atol=1e-12)
        np.testing.assert_allclose(camera[:3,:3]@poses[1][:3,:3],Rotation.from_rotvec([0,0,.2]).as_matrix(),atol=1e-12)
        np.testing.assert_array_equal(seed,before)

    def test_arbitrary_face_flip_recovers_prior_orientation_without_freezing_translation(self):
        mesh=trimesh.creation.box(extents=[.2,.12,.01]);mesh.apply_translation([.1,-.03,.02])
        renderer=MeshSilhouette(mesh.vertices,mesh.faces,dict(fx=700.,fy=700.,cx=512.,cy=384.))
        prior=np.eye(4);prior[:3,:3]=Rotation.from_rotvec([.2,-.1,.3]).as_matrix()
        axis=np.array([np.cos(.37),np.sin(.37),0.])
        seed=prior.copy();seed[:3,:3]=prior[:3,:3]@Rotation.from_rotvec(axis*np.pi).as_matrix();seed[:3,3]=[.15,-.06,.7]
        poses=moving_orientation_proposals(renderer,seed,np.eye(4),prior)
        np.testing.assert_allclose(poses[-1][:3,:3],prior[:3,:3],atol=1e-12)
        self.assertGreater(np.linalg.norm(poses[-1][:3,3]-prior[:3,3]),.5)

    def test_two_sided_pose_hypotheses_preserve_mesh_centre_and_handedness(self):
        mesh=trimesh.creation.box(extents=[.2,.12,.01]);mesh.apply_translation([.1,-.03,.02])
        renderer=MeshSilhouette(mesh.vertices,mesh.faces,dict(fx=700.,fy=700.,cx=512.,cy=384.))
        seed=np.eye(4);seed[:3,3]=[.05,-.01,.65];center=mesh.vertices.mean(axis=0)
        for M in face_flip_proposals(renderer,seed):
            np.testing.assert_allclose(M[:3,:3]@center+M[:3,3],seed[:3,:3]@center+seed[:3,3],atol=1e-9)
            self.assertAlmostEqual(np.linalg.det(M[:3,:3]),1.)

    def test_smooth_rotation_is_not_penalized_as_an_instant_face_flip(self):
        earlier=np.eye(4);previous=np.eye(4);previous[:3,:3]=Rotation.from_rotvec([0,0,.08]).as_matrix()
        expected=np.eye(4);expected[:3,:3]=Rotation.from_rotvec([0,0,.16]).as_matrix()
        self.assertLess(temporal_pose_cost(expected,previous,earlier),1e-12)
        flipped=expected.copy();flipped[:3,:3]=Rotation.from_rotvec([0,0,np.pi]).as_matrix()@expected[:3,:3]
        self.assertAlmostEqual(temporal_pose_cost(flipped,previous,earlier),.1)
    def test_asymmetric_rigid_object_pose_recovers_from_camera_frame_error(self):
        body=trimesh.creation.box(extents=[.16,.10,.05]);handle=trimesh.creation.box(extents=[.14,.025,.02]);handle.apply_translation([.14,.02,0.])
        mesh=trimesh.util.concatenate([body,handle]);intr=dict(fx=700.,fy=700.,cx=512.,cy=384.,width=1024,height=768)
        renderer=MeshSilhouette(mesh.vertices,mesh.faces,intr);truth=np.eye(4);truth[:3,:3]=Rotation.from_rotvec([.2,-.1,.2]).as_matrix();truth[2,3]=.65
        target=renderer.render(truth);depth=renderer.depth(truth);seed=truth.copy();seed[:3,3]+=[.009,-.005,.012];seed[:3,:3]=Rotation.from_rotvec([.025,-.035,.06]).as_matrix()@seed[:3,:3]
        initial=renderer.render(seed);before=(initial&target).sum()/(initial|target).sum()
        fitted=propose(renderer,seed,target,np.ones_like(target),intr,depth,rounds=5)
        self.assertIsNotNone(fitted)
        result=renderer.render(fitted);after=(result&target).sum()/(result|target).sum()
        self.assertGreater(after,before+.08)
        self.assertLess(np.linalg.norm(fitted[:3,3]-truth[:3,3]),.006)

    def test_unobservable_source_boundary_never_generates_a_pose(self):
        class Renderer:pass
        self.assertIsNone(propose(Renderer(),np.eye(4),np.zeros((384,512),bool),np.ones((384,512),bool),dict(fx=700.,fy=700.,cx=512.,cy=384.)))
