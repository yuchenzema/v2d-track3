import unittest
from types import SimpleNamespace
from unittest.mock import patch
import numpy as np
from native_pinch_contacts import valid_distance_witness,NativePinchContacts


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

    def test_touching_witness_uses_only_the_exact_native_contact_and_correct_direction(self):
        for pair,expected in [([1,2],[1.,0.,0.]),([2,1],[-1.,0.,0.])]:
            pinch=NativePinchContacts.__new__(NativePinchContacts)
            pinch.pairs=[('index',1,2)]
            pinch.model=SimpleNamespace(geom_friction=np.ones((3,3)))
            pinch.data=SimpleNamespace(contact=[SimpleNamespace(geom=pair,frame=np.eye(3),pos=np.array([.1,.2,.3]))])
            with patch.dict('sys.modules',{'mujoco':SimpleNamespace(mj_geomDistance=lambda *args:0.)}):
                row=pinch.query()[0]
            self.assertEqual(row['normal_on_object'],expected)
            self.assertEqual(row['position'],[.1,.2,.3]);self.assertTrue(row['valid_witness'])
