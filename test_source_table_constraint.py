import unittest
import numpy as np
from source_table_constraint import constrain_pose

class TableConstraintTests(unittest.TestCase):
    def test_intersecting_geometry_is_lifted_without_changing_rotation(self):
        pose=np.eye(4);pose[2,3]=.62
        result=constrain_pose(pose,np.array([[0,0,-.02],[.01,0,.1]]),[0,0,.63],.23,.6505)
        self.assertAlmostEqual(result[2,3],.6705)
        np.testing.assert_array_equal(result[:3,:3],pose[:3,:3])
        self.assertEqual(pose[2,3],.62)
    def test_object_outside_table_can_move_below_surface(self):
        pose=np.eye(4);pose[:3,3]=[1,0,.5]
        np.testing.assert_array_equal(constrain_pose(pose,np.zeros((1,3)),[0,0,0],.23,.65),pose)
    def test_lifted_object_is_not_pulled_down_during_grasp(self):
        pose=np.eye(4);pose[2,3]=1.
        np.testing.assert_array_equal(constrain_pose(pose,np.zeros((1,3)),[0,0,0],.23,.65),pose)
        self.assertAlmostEqual(constrain_pose(pose,np.zeros((1,3)),[0,0,0],.23,.65,settled=True)[2,3],.65)

if __name__=='__main__':unittest.main()
