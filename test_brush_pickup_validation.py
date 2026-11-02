import unittest
import numpy as np
from assess_episode30_pickup import assess_arrays

class PhysicalPickupTests(unittest.TestCase):
    def setup_motion(self):
        f=np.linspace(0,262,524);obj=np.zeros((524,7));obj[:,6]=1
        obj[:,0]=np.clip((f-34)/30,0,1)*.08;obj[:,2]=.73+np.clip((f-34)/20,0,1)*.04
        palm=obj.copy();palm[:,1]+=.08
        return f,obj,palm,obj[:,:3].copy()
    def test_held_brush_following_correct_trajectory_passes_pickup_only(self):
        f,o,p,r=self.setup_motion();result=assess_arrays(o,p,f,r,np.zeros(524))
        self.assertTrue(result['passes']);self.assertFalse(result['exact_human_motion_match_verified'])
    def test_moving_hand_with_stationary_brush_fails(self):
        f,o,p,r=self.setup_motion();o[:,:3]=o[0,:3]
        self.assertFalse(assess_arrays(o,p,f,r,np.zeros(524))['passes'])
    def test_lift_without_sustained_hold_fails(self):
        f,o,p,r=self.setup_motion();p[f>90,0]+=.1
        self.assertFalse(assess_arrays(o,p,f,r,np.zeros(524))['passes'])
    def test_assisted_object_cannot_pass(self):
        f,o,p,r=self.setup_motion()
        self.assertFalse(assess_arrays(o,p,f,r,np.ones(524))['passes'])
    def test_missing_pickup_frames_cannot_pass(self):
        f,o,p,r=self.setup_motion();keep=f<40
        self.assertFalse(assess_arrays(o[keep],p[keep],f[keep],r[keep],np.zeros(keep.sum()))['passes'])

if __name__=='__main__':unittest.main()
