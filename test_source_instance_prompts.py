import unittest
import numpy as np
from source_instance_prompts import interior_points,separated_frame


class SourceInstancePromptTests(unittest.TestCase):
    def test_large_merged_masks_cannot_win_over_separate_measured_instances(self):
        a=np.zeros((80,100),bool);a[10:65,5:40]=True
        b=np.zeros_like(a);b[15:60,60:90]=True
        _,frame,masks=separated_frame([(0,[a|b,a|b]),(150,[a,b])])
        self.assertEqual(frame,150)
        self.assertFalse(np.any(masks[0]&masks[1]))
        for mask in masks:
            for point in interior_points(mask):self.assertTrue(mask[int(point['y']),int(point['x'])])

    def test_no_separate_source_evidence_cannot_invent_instance_hints(self):
        mask=np.ones((30,30),bool)
        with self.assertRaisesRegex(ValueError,'No source view'):separated_frame([(0,[mask,mask])])
