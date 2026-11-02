import unittest
from grasp_physics_checks import gravity_cone_feasible,gravity_cone_solution
import numpy as np


class FrictionConeTests(unittest.TestCase):
    def test_returned_forces_balance_gravity_and_moment(self):
        points=np.array([[-.05,0,0],[.05,0,0]])
        result=gravity_cone_solution([[1,0,0],[-1,0,0]],points,[.8,.8],1.,[0,0,0])
        self.assertIsNotNone(result)
        np.testing.assert_allclose(result['forces_N'].sum(axis=0),[0,0,9.81],atol=1e-6)
        np.testing.assert_allclose(np.cross(points,result['forces_N']).sum(axis=0),0,atol=1e-6)
        self.assertTrue(np.all(result['normal_loads_N']<=25.))

    def test_opposed_frictional_fingers_support_weight(self):
        self.assertTrue(gravity_cone_feasible([[1,0,0],[-1,0,0]],[[-.05,0,0],[.05,0,0]],[.8,.8],1.,[0,0,0]))

    def test_frictionless_horizontal_contacts_cannot_lift(self):
        self.assertFalse(gravity_cone_feasible([[1,0,0],[-1,0,0]],[[-.05,0,0],[.05,0,0]],[0.,0.],1.,[0,0,0]))

    def test_one_sided_horizontal_contact_cannot_balance(self):
        self.assertFalse(gravity_cone_feasible([[1,0,0]],[[-.05,0,0]],[.8],1.,[0,0,0]))

    def test_force_limit_rejects_excessive_weight(self):
        self.assertFalse(gravity_cone_feasible([[1,0,0],[-1,0,0]],[[-.05,0,0],[.05,0,0]],[.8,.8],100.,[0,0,0]))


if __name__=='__main__':unittest.main()
