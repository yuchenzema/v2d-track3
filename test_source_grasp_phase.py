import unittest
from source_grasp_phase import template_frames


class SupportPhaseTests(unittest.TestCase):
    def test_handover_examples_do_not_replace_the_required_single_hand_grasp(self):
        grasps=[dict(side='right',object=1,start=27,end=61),dict(side='left',object=1,start=41,end=66),dict(side='right',object=1,start=66,end=134)]
        self.assertEqual(template_frames(grasps,grasps[0]),(list(range(35,41)),'sole_hand_support'))
        self.assertEqual(template_frames(grasps,grasps[1]),(list(range(62,66)),'sole_hand_support'))
        self.assertTrue(all(frame>66 for frame in template_frames(grasps,grasps[2])[0]))

    def test_true_two_hand_support_and_another_object_remain_distinct(self):
        grasp=dict(side='right',object=0,start=10,end=30)
        both=[grasp,dict(side='left',object=0,start=5,end=40)]
        self.assertEqual(template_frames(both,grasp),(list(range(15,26)),'cooperative_support_only'))
        both[1]['object']=1
        self.assertEqual(template_frames(both,grasp)[1],'sole_hand_support')
