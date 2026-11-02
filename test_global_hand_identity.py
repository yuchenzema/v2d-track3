import unittest
from global_hand_identity import decide


class WholeTrackHandTests(unittest.TestCase):
    def test_majority_vote_ignores_isolated_per_frame_label_flips(self):
        permutation,confidence=decide([[80,20],[15,85]],[100,100]);self.assertEqual(permutation,[0,1]);self.assertGreaterEqual(min(confidence),.8)

    def test_initially_swapped_tracks_are_corrected_once_for_whole_sequence(self):
        permutation,_=decide([[5,90],[88,7]],[95,95]);self.assertEqual(permutation,[1,0])

    def test_ambiguous_anatomy_is_blocked(self):
        with self.assertRaises(RuntimeError):decide([[51,49],[48,52]],[100,100])
