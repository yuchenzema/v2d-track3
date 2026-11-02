import unittest
import numpy as np
from scipy.spatial.transform import Rotation
from source_cad_alignment import diverse_pose_seeds,source_centroid_cost


class PoseSeedTests(unittest.TestCase):
    def test_source_centroid_cost_is_neutral_inside_gate_and_penalizes_drift(self):
        mask=np.zeros((30,30),bool);mask[10:20,10:20]=True
        center=np.array([14.5,14.5])
        self.assertEqual(source_centroid_cost(mask,center),0.)
        self.assertEqual(source_centroid_cost(np.roll(mask,1,axis=1),center),0.)
        close=source_centroid_cost(np.roll(mask,3,axis=1),center)
        far=source_centroid_cost(np.roll(mask,6,axis=1),center)
        self.assertGreater(close,0.)
        self.assertGreater(far,close)
        self.assertEqual(source_centroid_cost(np.zeros_like(mask),center),1.)

    def test_duplicate_basin_does_not_hide_rotated_candidate(self):
        original=np.eye(4);near=original.copy();near[0,3]=.001
        rotated=original.copy();rotated[:3,:3]=Rotation.from_rotvec([0,0,1]).as_matrix()
        translated=original.copy();translated[0,3]=.02
        got=diverse_pose_seeds([original,near,original,rotated,translated],3)
        self.assertEqual(len(got),3)
        np.testing.assert_array_equal(got[1],rotated)
        np.testing.assert_array_equal(got[2],translated)

    def test_nonfinite_and_empty_proposals(self):
        invalid=np.eye(4);invalid[0,0]=np.nan
        self.assertEqual(diverse_pose_seeds([invalid]),[])
        self.assertEqual(diverse_pose_seeds([]),[])
