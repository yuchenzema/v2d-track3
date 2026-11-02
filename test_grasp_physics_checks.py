import unittest
from grasp_physics_checks import gravity_cone_feasible


class FrictionConeTests(unittest.TestCase):
    def test_opposed_frictional_fingers_support_weight(self):
        self.assertTrue(gravity_cone_feasible([[1,0,0],[-1,0,0]],[[-.05,0,0],[.05,0,0]],[.8,.8],1.,[0,0,0]))

    def test_frictionless_horizontal_contacts_cannot_lift(self):
        self.assertFalse(gravity_cone_feasible([[1,0,0],[-1,0,0]],[[-.05,0,0],[.05,0,0]],[0.,0.],1.,[0,0,0]))

    def test_one_sided_horizontal_contact_cannot_balance(self):
        self.assertFalse(gravity_cone_feasible([[1,0,0]],[[-.05,0,0]],[.8],1.,[0,0,0]))

    def test_force_limit_rejects_excessive_weight(self):
        self.assertFalse(gravity_cone_feasible([[1,0,0],[-1,0,0]],[[-.05,0,0],[.05,0,0]],[.8,.8],100.,[0,0,0]))


if __name__=='__main__':unittest.main()
