"""Run with the installed native MuJoCo CPU backend; no training or GPU."""
import unittest
import mujoco
import numpy as np
from rigid_pose_collision_audit import PoseCollisionAuditor,hand_clearance_penalties


class CollisionReplayTests(unittest.TestCase):
    def setUp(self):
        xml = '''<mujoco><worldbody><geom type="plane" size="10 10 .1"/>
        <body name="left_hand" pos="0 0 1"><freejoint/><geom type="sphere" size=".05"/></body>
        <body name="right_hand" pos="2 0 1"><freejoint/><geom type="sphere" size=".05"/></body>
        <body name="track3_cup_object" pos=".3 0 1"><freejoint/><geom type="sphere" size=".05"/></body>
        <body name="track3_spoon_object" pos="3 0 1"><freejoint/><geom type="sphere" size=".05"/></body>
        </worldbody></mujoco>'''
        self.model = mujoco.MjModel.from_xml_string(xml); self.data = mujoco.MjData(self.model)
        mujoco.mj_forward(self.model, self.data)
        self.names = ['left_hand', 'right_hand', 'track3_cup/object', 'track3_spoon/object']
        self.poses = np.c_[self.data.xpos[1:], self.data.xquat[1:, [1, 2, 3, 0]]]
        self.auditor = PoseCollisionAuditor(self.model, self.data, self.names, self.poses, ['cup', 'spoon'])

    def test_separated_solids_pass(self):
        row = self.auditor.inspect(self.poses)
        self.assertEqual(row['penetrating_pair_count'], 0)
        self.assertEqual(len(row['maximum_by_category_m']), 5)

    def test_hand_object_penetration_is_measured(self):
        self.poses[2, 0] = .09
        row = self.auditor.inspect(self.poses)
        self.assertAlmostEqual(row['maximum_penetration_m'], .01, places=7)
        self.assertEqual(row['worst_contacts'][0]['category'], 'hand_object')

    def test_object_object_and_environment_are_checked(self):
        self.poses[3, :3] = self.poses[2, :3] + [.09, 0, 0]
        self.assertAlmostEqual(self.auditor.inspect(self.poses)['maximum_by_category_m']['object_object'], .01)
        self.poses[2, 2] = .04
        self.assertAlmostEqual(self.auditor.inspect(self.poses)['maximum_by_category_m']['object_environment'], .01)

    def test_pair_query_does_not_hide_disabled_interactions(self):
        self.model.geom_contype[:] = 0; self.model.geom_conaffinity[:] = 0
        self.poses[2, 0] = .09
        self.assertAlmostEqual(self.auditor.inspect(self.poses)['maximum_penetration_m'], .01)

    def test_grasp_search_focus_cannot_hide_target_hand_overlap(self):
        self.poses[2, 0] = .09
        self.poses[3, 0] = 2.06
        whole=self.auditor.inspect(self.poses)
        focused=self.auditor.inspect(self.poses,focus_group='hand:left')
        self.assertAlmostEqual(whole['maximum_penetration_m'],.04)
        self.assertAlmostEqual(focused['maximum_penetration_m'],.01)
        self.assertLess(focused['pairs_covered'],whole['pairs_covered'])

    def test_nonfinite_poses_and_missing_body_fail(self):
        self.poses[0, 0] = np.nan
        with self.assertRaises(AssertionError): self.auditor.inspect(self.poses)
        self.poses[0, 0] = 0.
        with self.assertRaises(AssertionError):
            PoseCollisionAuditor(self.model, self.data, self.names[:-1], self.poses[:-1], ['cup', 'spoon'])

    def test_reference_fitter_detects_solver_excluded_hand_table_overlap(self):
        model=mujoco.MjModel.from_xml_string('''<mujoco><worldbody>
          <body name="hand" pos="0 0 .04"><freejoint/><geom type="sphere" size=".05"/></body>
          <body name="support" pos="0 0 -.05"><geom type="box" size="1 1 .05"/></body>
          </worldbody><contact><exclude body1="hand" body2="support"/></contact></mujoco>''')
        data=mujoco.MjData(model);mujoco.mj_forward(model,data)
        self.assertEqual(data.ncon,0)
        penalties,maximum=hand_clearance_penalties(model,data,[0],{1})
        self.assertAlmostEqual(maximum,.01,places=7)
        self.assertAlmostEqual(penalties[0],.0105,places=7)


if __name__ == '__main__':
    unittest.main()
