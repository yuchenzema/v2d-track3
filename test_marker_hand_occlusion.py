import unittest
from unittest.mock import patch
import numpy as np
from sphere_marker_reference import measure_markers


class MarkerHandOcclusionTests(unittest.TestCase):
    def scene(self):
        image=np.zeros((100,100,3),np.uint8)
        template=dict(markers=[dict(center_m=[0.,0.,0.],radius_m=.01),dict(center_m=[.2,0.,0.],radius_m=.01)])
        pose=np.eye(4);pose[2,3]=1.
        return image,template,pose,dict(fx=100.,fy=100.,cx=50.,cy=50.)

    @staticmethod
    def candidate(xy):return dict(xy=list(xy),radius_px=1.,cost=.1)

    def test_hand_occluded_prediction_cannot_match_an_unrelated_circle_outside_the_hand(self):
        image,template,pose,intr=self.scene();hand=np.zeros((100,100),bool);hand[45:56,45:56]=True
        def circles(image,prediction,radius):
            return [self.candidate([58,50] if prediction[0]<60 else [70,50])]
        with patch('sphere_marker_reference.circle_candidates',side_effect=circles):
            observations=measure_markers(image,template,pose,intr,hand_occlusion=hand)
        self.assertFalse(observations[0]['observed']);self.assertEqual(observations[0]['visibility'],'hand_occluded')
        self.assertTrue(observations[1]['observed']);self.assertEqual(observations[1]['measurement']['xy'],[70,50])

    def test_unoccluded_marker_measurements_are_preserved(self):
        image,template,pose,intr=self.scene()
        with patch('sphere_marker_reference.circle_candidates',side_effect=lambda image,p,r:[self.candidate(p)]):
            observations=measure_markers(image,template,pose,intr,hand_occlusion=np.zeros((100,100),bool))
        self.assertTrue(all(x['observed'] for x in observations))

    def test_candidate_inside_hand_remains_rejected_even_when_prediction_is_outside(self):
        image,template,pose,intr=self.scene();hand=np.zeros((100,100),bool);hand[45:56,45:56]=True
        with patch('sphere_marker_reference.circle_candidates',return_value=[self.candidate([50,50])]):
            observations=measure_markers(image,template,pose,intr,hand_occlusion=hand)
        self.assertFalse(any(x['observed'] for x in observations))
