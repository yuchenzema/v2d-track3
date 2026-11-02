import unittest
from unittest.mock import patch
from pathlib import Path
import tempfile
import numpy as np
from source_cad_alignment import metrics,qualifies,save_trajectory,choose_qualified_source_pose
from source_mask_rasterizer import MeshSilhouette
from sphere_marker_reference import marker_residual
import trimesh


class SourceCadAlignmentTests(unittest.TestCase):
    def source_pose_fixture(self):
        mesh=trimesh.creation.box(extents=[.12,.20,.01])
        intr=dict(fx=700.,fy=700.,cx=512.,cy=384.)
        renderer=MeshSilhouette(mesh.vertices,mesh.faces,intr)
        truth=np.eye(4);truth[2,3]=.6;target=renderer.render(truth)
        xyz=[[.03,.04,0.]];observed=marker_residual(truth,xyz,[[0.,0.]],intr)
        def measure(pose):
            row=metrics(renderer.render(pose),target,np.ones(target.shape,bool))
            row['sphere_error_px']=float(np.linalg.norm(marker_residual(pose,xyz,observed,intr)))
            return row
        return truth,measure

    def test_soft_objective_cannot_replace_available_source_valid_pose_with_shifted_pose(self):
        truth,measure=self.source_pose_fixture();bad=truth.copy();bad[0,3]=.008
        self.assertFalse(qualifies(measure(bad)))
        pose,_,row,info=choose_qualified_source_pose([truth],bad,0.,measure,lambda p:0. if p[0,3] else 1.)
        np.testing.assert_array_equal(pose,truth);self.assertTrue(qualifies(row));self.assertTrue(info['replaced_unqualified_best'])

    def test_silhouette_equivalent_face_with_wrong_image_caps_remains_ineligible(self):
        truth,measure=self.source_pose_fixture();wrong=truth.copy();wrong[:2,:2]*=-1
        self.assertGreater(measure(wrong)['iou'],.99)
        self.assertFalse(qualifies(measure(wrong)))
        pose,_,row,_=choose_qualified_source_pose([truth],wrong,0.,measure,lambda p:0. if p[0,0]<0 else 1.)
        np.testing.assert_array_equal(pose,truth);self.assertTrue(qualifies(row))

    def test_no_source_valid_candidate_retains_failure_instead_of_certifying_a_compromise(self):
        truth,measure=self.source_pose_fixture();bad=truth.copy();bad[0,3]=.04;other=truth.copy();other[0,3]=.05
        pose,_,row,info=choose_qualified_source_pose([other],bad,0.,measure,lambda p:1.)
        np.testing.assert_array_equal(pose,bad);self.assertFalse(qualifies(row));self.assertEqual(info['qualified_candidates'],0)

    def test_existing_objective_still_ranks_candidates_that_all_meet_source_gates(self):
        truth,measure=self.source_pose_fixture();near=truth.copy();near[0,3]=.0005;bad=truth.copy();bad[0,3]=.04
        pose,_,row,info=choose_qualified_source_pose([truth,near,near],bad,-1.,measure,lambda p:1. if p[0,3] else 2.)
        np.testing.assert_array_equal(pose,near);self.assertTrue(qualifies(row));self.assertEqual(info['qualified_candidates'],2)

    def test_failed_checkpoint_write_keeps_previous_trajectory_resumable(self):
        with tempfile.TemporaryDirectory() as directory:
            path=Path(directory)/'trajectory.npz'
            save_trajectory(path,{'object_to_world':np.eye(4)[None,None]})
            before=path.read_bytes()
            with patch('source_cad_alignment.np.savez_compressed',side_effect=OSError('disk full')):
                with self.assertRaises(OSError):save_trajectory(path,{'object_to_world':np.zeros((1,1,4,4))})
            self.assertEqual(path.read_bytes(),before)
            np.testing.assert_array_equal(np.load(path)['object_to_world'],np.eye(4)[None,None])
            self.assertEqual(list(Path(directory).iterdir()),[path])

    def test_perfect_visible_mask_passes(self):
        mask=np.zeros((100,100),bool);mask[20:60,30:70]=True
        row=metrics(mask,mask,np.ones_like(mask))
        self.assertEqual(row['iou'],1.)
        self.assertTrue(qualifies(row))

    def test_occlusion_cannot_be_claimed_done(self):
        mask=np.zeros((100,100),bool);mask[20:60,30:70]=True
        row=metrics(mask,mask,np.zeros_like(mask))
        self.assertFalse(qualifies(row))

    def test_shifted_trajectory_fails_even_if_geometry_is_identical(self):
        mask=np.zeros((100,100),bool);mask[20:60,30:70]=True
        shifted=np.roll(mask,10,axis=1)
        row=metrics(shifted,mask,np.ones_like(mask))
        self.assertGreater(row['centroid_error_px'],3.)
        self.assertFalse(qualifies(row))

    def test_bad_sphere_alignment_blocks_silhouette_pass(self):
        mask=np.zeros((100,100),bool);mask[20:60,30:70]=True
        row=metrics(mask,mask,np.ones_like(mask));row['sphere_error_px']=8.
        self.assertFalse(qualifies(row))


if __name__=='__main__':unittest.main()
