import unittest
import numpy as np
from post_training_grasp_acceptance import score
from episode_completion_gate import completion


class PostTrainingGraspTests(unittest.TestCase):
    def test_source_following_policy_passes_but_stationary_object_fails(self):
        reference=np.zeros((30,1,3));reference[:,0,2]=np.linspace(0,.17,30)
        grasps=[dict(start=0,end=29,object=0)]
        self.assertTrue(score(reference,reference.copy(),grasps)['all_grasps_passed'])
        self.assertFalse(score(reference,np.zeros_like(reference),grasps)['all_grasps_passed'])

    def test_incomplete_or_nonfinite_recording_cannot_pass(self):
        reference=np.zeros((30,1,3));actual=reference.copy();actual[0,0,0]=np.nan
        with self.assertRaises(AssertionError):score(reference,actual,[dict(start=0,end=29,object=0)])
        with self.assertRaises(AssertionError):score(reference,reference[:-1],[dict(start=0,end=29,object=0)])

    def test_final_gate_rejects_missing_or_unbound_policy_grasp_evidence(self):
        result=completion(dict(post_training_grasp_acceptance_required=True))
        self.assertTrue(any('Trained-policy grasp' in r for r in result['blocking_reasons']))


if __name__=='__main__':unittest.main()
