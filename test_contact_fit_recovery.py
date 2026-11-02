import unittest
from rigid_pose_collision_audit import contact_retry_thresholds,unfitted_blend_frames


class ContactFitRecoveryTests(unittest.TestCase):
    def test_native_episode38_millimetre_collisions_trigger_strict_rescue(self):
        opening,escape=contact_retry_thresholds(True)
        for depth in [.0029017700605998543,.002687166996367925,.000002]:
            self.assertGreater(depth,opening)
            self.assertGreater(depth,escape)
        self.assertLessEqual(.000001,opening)
        self.assertEqual(contact_retry_thresholds(False),(.003,.005))

    def test_smoothing_never_overwrites_any_explicit_collision_fit(self):
        first={'held_interval':[31,31]};second={'held_interval':[32,32]}
        fits=[first,second,{'held_interval':[33,33]}]
        lo,hi,frames=unfitted_blend_frames(first,second,fits,263)
        self.assertEqual(frames,[])
        self.assertEqual((lo,hi),(31,32))

    def test_blending_only_updates_unfitted_frames_within_source_grid(self):
        first={'held_interval':[0,0]};second={'held_interval':[1,5]}
        lo,hi,frames=unfitted_blend_frames(first,second,[{'held_interval':[0,0]},{'held_interval':[3,3]}],4)
        self.assertEqual(frames,[1,2])


if __name__=='__main__':unittest.main()
