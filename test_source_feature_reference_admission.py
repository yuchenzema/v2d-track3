import unittest

from source_cad_alignment import source_feature_reference_available


class SourceFeatureReferenceAdmissionTests(unittest.TestCase):
    def test_white_pot_can_use_the_immediately_previous_verified_source_surface(self):
        self.assertTrue(source_feature_reference_available('white_pot',135,[
            dict(frame=134,object='white_pot',local_overlap_pass=True)]))

    def test_failed_stale_missing_or_other_object_evidence_cannot_enable_tracking(self):
        for rows in [[],[dict(frame=133,object='white_pot',local_overlap_pass=True)],
                     [dict(frame=134,object='blue_cup',local_overlap_pass=True)],
                     [dict(frame=134,object='white_pot',local_overlap_pass=False)],
                     [dict(frame=134,object='white_pot',local_overlap_pass='true')],
                     [dict(frame=134,object='white_pot',local_overlap_pass=True),
                      dict(frame=134,object='white_pot',local_overlap_pass=False)]]:
            with self.subTest(rows=rows):
                self.assertFalse(source_feature_reference_available('white_pot',135,rows))

    def test_frame_zero_and_unvalidated_object_types_do_not_inherit_a_surface(self):
        self.assertFalse(source_feature_reference_available('white_pot',0,[]))
        self.assertFalse(source_feature_reference_available('blue_cup',2,[
            dict(frame=1,object='blue_cup',local_overlap_pass=True)]))

    def test_existing_wooden_board_tracking_remains_available(self):
        for name in ['wooden_piece_1','wooden_piece_2']:
            self.assertTrue(source_feature_reference_available(name,1,[]))
