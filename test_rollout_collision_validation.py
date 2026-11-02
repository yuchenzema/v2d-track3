import copy
import unittest
from rollout_collision_validation import collision_verified


def passing_collision():
    return {'all_recorded_frames_checked': True, 'all_required_pairs_checked': True,
            'no_penetration_verified': True, 'failed_frame_count': 0,
            'states_sha256': 'recorded-state', 'version': 'video-v1', 'recorded_frame_count': 402,
            'geometry_sha256': 'scene-geometry', 'maximum_penetration_m': 0., 'numerical_epsilon_m': 1e-6}


class CollisionProofTests(unittest.TestCase):
    def test_complete_bound_report_passes(self):
        self.assertTrue(collision_verified(passing_collision(), 'recorded-state', 'video-v1', 402))

    def test_untested_stale_and_partial_evidence_fails_closed(self):
        self.assertFalse(collision_verified({}, 'recorded-state'))
        for key, value in [('states_sha256', 'another'), ('version', 'another'),
                           ('recorded_frame_count', 401), ('geometry_sha256', ''),
                           ('all_recorded_frames_checked', False), ('all_required_pairs_checked', False)]:
            report = passing_collision(); report[key] = value
            self.assertFalse(collision_verified(report, 'recorded-state', 'video-v1', 402), key)

    def test_penetrations_cannot_be_accepted_by_loosening_tolerance(self):
        for maximum, epsilon in [(.001, 1e-6), (.001, .002), (float('nan'), 1e-6), (0., float('inf'))]:
            report = passing_collision(); report.update(maximum_penetration_m=maximum, numerical_epsilon_m=epsilon)
            self.assertFalse(collision_verified(report, 'recorded-state'))

    def test_report_cannot_claim_pass_with_failed_frames(self):
        report = passing_collision(); report['failed_frame_count'] = 1
        self.assertFalse(collision_verified(report, 'recorded-state'))
