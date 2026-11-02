"""Checks that visual overlap alone cannot hide orientation or trajectory failures."""
import copy
import unittest
import math
import numpy as np
from scipy.spatial.transform import Rotation
from human_match_criteria import assess
from source_pose_seed import rebase_reconstructed_pose

def fixture():
    obj={'rollout':{'iou':1.,'centroid_error_px':0.,'visible_source_pixels':100},
         'reconstruction':{'iou':1.,'centroid_error_px':0.},
         'object_position_error_to_video_reconstruction_m':0.,'object_orientation_error_to_video_reconstruction_rad':0.,
         'rollout_position_world_m':[0.,0.,0.],'reconstruction_position_world_m':[0.,0.,0.],
         'rollout_orientation_world_xyzw':[0.,0.,0.,1.],'reconstruction_orientation_world_xyzw':[0.,0.,0.,1.]}
    return {'episode':19,'version':'fixture','source_frames':2,'objects':{'brush':{}}},{'frames':[{'frame':i,'objects':{'brush':copy.deepcopy(obj)}} for i in range(2)]}

class CorrespondenceTests(unittest.TestCase):
    def test_perfect_video_overlap_does_not_hide_penetration(self):
        summary,data=fixture();summary['collision_validation']={'all_recorded_frames_checked':True,
            'failed_frame_count':1,'recorded_frame_count':2,'failed_control_frames':[1]}
        result=assess(summary,data)
        self.assertFalse(result['passes_strict_local_checks'])
        self.assertTrue(result['contact_validation_failed'])
        self.assertEqual(result['contact_failure_source_frames'],[1])
    def test_contact_failure_does_not_stall_in_visibility_only_review(self):
        summary,data=fixture()
        for frame in data['frames']:frame['objects']['brush']['rollout']['visible_source_pixels']=0
        summary['collision_validation']={'all_recorded_frames_checked':True,'failed_frame_count':1,
            'recorded_frame_count':2,'failed_control_frames':[1]}
        self.assertNotEqual(assess(summary,data)['recommended_action'],'visibility_review')
    def test_local_pass_never_certifies_exact_human_match(self):
        summary,data=fixture();result=assess(summary,data)
        self.assertTrue(result['passes_strict_local_checks']);self.assertFalse(result['exact_human_match_verified'])
    def test_perfect_mask_cannot_hide_wrong_orientation(self):
        summary,data=fixture();data['frames'][1]['objects']['brush']['object_orientation_error_to_video_reconstruction_rad']=.2
        self.assertEqual(assess(summary,data)['failed_frame_count'],1)
    def test_opposite_motion_is_detected_even_at_equal_speed(self):
        summary,data=fixture();v=data['frames'][1]['objects']['brush']
        v['rollout_position_world_m']=[.004,0,0];v['reconstruction_position_world_m']=[-.004,0,0]
        v['object_position_error_to_video_reconstruction_m']=.008
        result=assess(summary,data);self.assertEqual(result['failed_frame_count'],1)
        self.assertAlmostEqual(result['objects']['brush']['per_frame_errors'][1]['trajectory_step_error_m'],.008)
    def test_hidden_frames_remain_unverified(self):
        summary,data=fixture();data['frames'][1]['objects']['brush']['rollout']['visible_source_pixels']=0
        result=assess(summary,data);self.assertFalse(result['passes_strict_local_checks'])
        self.assertEqual(result['recommended_action'],'visibility_review')
    def test_missing_source_frame_is_rejected(self):
        summary,data=fixture();data['frames'].pop()
        with self.assertRaises(AssertionError):assess(summary,data)
    def test_frame_data_from_another_release_are_rejected(self):
        summary,data=fixture();data['version']='old-video'
        with self.assertRaises(AssertionError):assess(summary,data)
    def test_shifted_timing_is_rejected(self):
        summary,data=fixture();data['frames'][1]['time_s']=.1
        with self.assertRaises(AssertionError):assess(summary,data)
    def test_nonfinite_pose_error_is_not_a_pass(self):
        summary,data=fixture();data['frames'][1]['objects']['brush']['object_position_error_to_video_reconstruction_m']=math.nan
        self.assertFalse(assess(summary,data)['passes_strict_local_checks'])
    def test_opposite_rotation_is_detected_between_frames(self):
        summary,data=fixture()
        for i,angle in enumerate([1.,-1.]):
            v=data['frames'][i]['objects']['brush'];v['object_orientation_error_to_video_reconstruction_rad']=math.radians(2)
            v['rollout_orientation_world_xyzw']=Rotation.from_euler('z',angle,degrees=True).as_quat().tolist()
            v['reconstruction_orientation_world_xyzw']=Rotation.from_euler('z',-angle,degrees=True).as_quat().tolist()
        self.assertEqual(assess(summary,data)['failed_frame_count'],1)
    def test_pose_seed_preserves_camera_view_and_object_dimensions(self):
        old=np.eye(4);old[:3,3]=[.2,.1,.8]
        new=np.eye(4);new[:3,:3]=Rotation.from_euler('z',45,degrees=True).as_matrix();new[:3,3]=[-.1,.3,.7]
        relative=np.eye(4);relative[:3,3]=[.1,.05,.4]
        result=rebase_reconstructed_pose(new,old,old@relative)
        points=np.array([[0,0,0,1],[.1,0,0,1],[0,.2,0,1]]).T
        camera_points=np.linalg.solve(new,result@points)
        self.assertTrue(np.allclose(camera_points,relative@points))
        self.assertAlmostEqual(float(np.linalg.norm((result@points)[:3,0]-(result@points)[:3,1])),.1)
        scaled=np.eye(4);scaled[0,0]=2.
        with self.assertRaises(ValueError):rebase_reconstructed_pose(new,old,scaled)

if __name__=='__main__':unittest.main()
