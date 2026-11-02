import copy
import unittest
from episode_completion_gate import completion,first_unfinished
from continuous_human_match_20261023 import execution_plan,required_episode
from test_rollout_collision_validation import passing_collision

def locally_passing():
    return {'all_frames_checked':True,'passes_strict_local_checks':True,'failed_frame_count':0,
            'unobservable_frame_count':0,'source_time_warping_used':False,'source_frames':202,
            'version':'video-v1','states_sha256':'recorded-state','original_video_sha256':'source-video','exact_human_match_verified':False}

def exact_evidence():
    result=locally_passing();result['exact_human_match_verified']=True
    result['collision_validation']=passing_collision()
    result['exact_match_evidence']={'version':'video-v1','states_sha256':'recorded-state',
        'all_objects_all_frames_verified':True,'source_frame_count':202,'source_video_sha256':'source-video',
        'method':'Independent exact-motion validation fixture'}
    return result

class CompletionGateTests(unittest.TestCase):
    def test_source_frame_policy_requires_exact_source_grid_collision_count(self):
        check=exact_evidence();check['source_frame_playback']=True
        self.assertFalse(completion(check)['episode_done'])
        check['collision_validation']['recorded_frame_count']=check['source_frames']
        self.assertTrue(completion(check)['episode_done'])
    def test_exact_motion_does_not_override_collision_failure(self):
        check=exact_evidence();check['collision_validation']['maximum_penetration_m']=.01
        self.assertFalse(completion(check)['episode_done'])
    def test_missing_collision_evidence_is_not_a_pass(self):
        check=exact_evidence();del check['collision_validation']
        self.assertFalse(completion(check)['advance_allowed'])
    def test_local_tolerances_are_not_exact_verification(self):
        self.assertFalse(completion(locally_passing())['advance_allowed'])
    def test_exact_claim_without_evidence_is_rejected(self):
        check=locally_passing();check['exact_human_match_verified']=True
        self.assertFalse(completion(check)['episode_done'])
    def test_every_object_and_frame_must_be_verified(self):
        for field in ['failed_frame_count','unobservable_frame_count']:
            check=exact_evidence();check[field]=1
            self.assertFalse(completion(check)['advance_allowed'])
    def test_stale_recording_or_video_evidence_is_rejected(self):
        for field in ['version','states_sha256']:
            check=exact_evidence();check['exact_match_evidence'][field]='another-artifact'
            self.assertFalse(completion(check)['advance_allowed'])
    def test_matching_full_evidence_can_release_next_episode(self):
        self.assertTrue(completion(exact_evidence())['advance_allowed'])
    def test_failed_episode20_blocks_later_episodes_even_if_they_pass(self):
        checks={'20':locally_passing(),'30':locally_passing()}
        self.assertEqual(first_unfinished([20,30,19],checks),20)
        state={'require_verified_match_before_next':True,'episodes':checks}
        self.assertEqual(required_episode(state),20)
        self.assertEqual(execution_plan(state,{'status':'completed'})['30']['status'],'waiting_for_current_episode_match')
        checks['20']=exact_evidence();self.assertEqual(first_unfinished([20,30,19],checks),30)
        checks['30']=exact_evidence();self.assertEqual(first_unfinished([20,30,19],checks),19)

if __name__=='__main__':unittest.main()
