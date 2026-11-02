import unittest
import numpy as np
from source_forearm_occlusion import measured_forearms


class ForearmEvidenceTests(unittest.TestCase):
    def scene(self):
        points=np.tile([512.,400.],(1,2,21,1))
        points[0,1,1]=[485,375]
        for index,x in zip([5,9,13,17],[480,500,525,545]):points[0,1,index]=[x,350]
        hands=dict(uv=points,observed=np.array([[False,True]]))
        depth=np.full((384,512),1.2);depth[165:384,235:285]=.5
        return hands,depth,np.zeros(depth.shape,bool)

    def test_only_supported_foreground_is_masked_and_object_pixels_are_protected(self):
        hands,depth,protected=self.scene();protected[220:230,250:260]=True
        depth[260:270,250:260]=0
        result=measured_forearms(hands,0,depth,protected).astype(bool)
        self.assertGreater(result.sum(),100)
        self.assertFalse(np.any(result&protected))
        self.assertFalse(np.any(result&(depth<=.05)))
        self.assertFalse(np.any(result&(depth>1.)))

    def test_absent_hand_or_missing_depth_cannot_create_forearm_evidence(self):
        hands,depth,protected=self.scene();hands['observed'][:]=False
        self.assertFalse(measured_forearms(hands,0,depth,protected).any())
        hands['observed'][0,1]=True;depth[:]=0
        self.assertFalse(measured_forearms(hands,0,depth,protected).any())
