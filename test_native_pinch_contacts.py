import unittest
import numpy as np
from native_pinch_contacts import valid_distance_witness


class DistanceWitnessTests(unittest.TestCase):
    def test_rejects_zero_distance_with_separated_witnesses(self):
        self.assertFalse(valid_distance_witness(0., [0, 0, 0, 0, 0, .2], .05))

    def test_accepts_consistent_separation_and_penetration(self):
        for distance in [.008, -.008]:
            self.assertTrue(valid_distance_witness(distance, [0, 0, 0, 0, 0, .008], .05))

    def test_rejects_no_hit_and_nonfinite_results(self):
        self.assertFalse(valid_distance_witness(.05, np.zeros(6), .05))
        self.assertFalse(valid_distance_witness(np.nan, np.zeros(6), .05))
        self.assertFalse(valid_distance_witness(.008, [0, 0, 0, np.nan, 0, 0], .05))
