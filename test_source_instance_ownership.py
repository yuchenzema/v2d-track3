import importlib.util
from pathlib import Path
import unittest

import numpy as np

source=Path(__file__).parent/'video_to_data/reconstruction/modules/v2d_sam2/lib/instance_masks.py'
spec=importlib.util.spec_from_file_location('source_instance_ownership',source)
module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module)
reconcile=module.exclusive_instance_masks
augment=module.cross_instance_negative_prompts


class SourceInstanceOwnershipTests(unittest.TestCase):
    def test_visible_foreground_is_negative_evidence_for_the_other_object(self):
        data=dict(prompts=[dict(object_id=1,frame_index=0,points=[dict(x=10,y=20)],point_labels=[1]),dict(object_id=2,frame_index=0,points=[dict(x=30,y=40)],point_labels=[1])])
        result,count=augment(data)
        self.assertEqual(count,2)
        self.assertEqual(result['prompts'][0]['points'][-1],dict(x=30,y=40))
        self.assertEqual(result['prompts'][0]['point_labels'],[1,0])
        self.assertEqual(data['prompts'][0]['point_labels'],[1])
        self.assertEqual(augment(result)[1],0)

    def test_constraints_never_invent_a_foreground_definition_on_another_frame(self):
        data=dict(prompts=[dict(object_id=1,frame_index=0,points=[dict(x=10,y=20)],point_labels=[1]),dict(object_id=2,frame_index=5,points=[dict(x=30,y=40)],point_labels=[1])])
        result,count=augment(data)
        self.assertEqual(count,0);self.assertEqual(result,data)

    def test_contradictory_visible_foreground_ownership_is_rejected(self):
        data=dict(prompts=[dict(object_id=i,frame_index=0,points=[dict(x=10,y=20)],point_labels=[1]) for i in [1,2]])
        with self.assertRaisesRegex(ValueError,'two object IDs'):augment(data)

    def test_visible_overlap_uses_measured_confidence_and_preserves_foreground_union(self):
        a=np.array([[4.,-2.,5.],[-3.,2.,-1.]])
        b=np.array([[1.,3.,7.],[-1.,-2.,-4.]])
        masks,count=reconcile({1:a,2:b})
        self.assertEqual(count,2)
        self.assertFalse(((masks[1]>0)&(masks[2]>0)).any())
        np.testing.assert_array_equal((masks[1]>0)|(masks[2]>0),(a>0)|(b>0))
        self.assertEqual(masks[1][0,0],255);self.assertEqual(masks[2][0,2],255)

    def test_no_positive_source_logit_never_creates_foreground(self):
        masks,count=reconcile({1:np.full((5,7),-2.),2:np.zeros((5,7))})
        self.assertEqual(count,0);self.assertFalse(any(mask.any() for mask in masks.values()))

    def test_non_overlapping_measurements_remain_unchanged(self):
        a=np.array([[2.,-1.],[-1.,-1.]])
        b=np.array([[-1.,4.],[-1.,-1.]])
        masks,count=reconcile({1:a,2:b})
        self.assertEqual(count,0)
        np.testing.assert_array_equal(masks[1],(a>0).astype('uint8')*255)
        np.testing.assert_array_equal(masks[2],(b>0).astype('uint8')*255)

    def test_ambiguous_tie_is_exclusive_and_independent_of_input_order(self):
        scores=np.ones((3,4))
        a,_=reconcile({2:scores,1:scores});b,_=reconcile({1:scores,2:scores})
        np.testing.assert_array_equal(a[1],b[1]);self.assertFalse(a[2].any())

    def test_invalid_measurements_fail_instead_of_inventing_ownership(self):
        with self.assertRaises(ValueError):reconcile({1:np.full((3,4),np.nan)})
        with self.assertRaises(ValueError):reconcile({1:np.ones((3,4)),2:np.ones((2,4))})
