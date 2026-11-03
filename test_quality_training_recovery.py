import contextlib,json,tempfile,unittest
from pathlib import Path
from unittest.mock import patch,Mock
import own_policy_quality_training as quality
from improvement_common import save


class QualityRecoveryTests(unittest.TestCase):
    def test_each_adopted_result_uploads_without_the_legacy_batch_or_count_limit(self):
        policy=dict(automatic_upload=True,upload_each_adopted_result=True)
        progress=dict(last_uploaded_reviewed_count=8,intermediate_upload_count=20)
        self.assertTrue(quality.should_upload(policy,progress,8,True,False))
        self.assertFalse(quality.should_upload(policy,progress,9,False,True))
        policy['automatic_upload']=False
        self.assertFalse(quality.should_upload(policy,progress,9,True,True))

    def test_legacy_upload_policy_still_honors_its_interval_and_limit(self):
        progress=dict(last_uploaded_reviewed_count=1,intermediate_upload_count=0)
        self.assertFalse(quality.should_upload({},progress,2,True,False))
        self.assertTrue(quality.should_upload({},progress,5,True,False))
        progress['intermediate_upload_count']=4
        self.assertFalse(quality.should_upload({},progress,5,True,True))

    def test_spider_success_is_considered_without_relaxing_the_source_image_gate(self):
        passed=dict(result=dict(local_reference_metrics=dict(spider_sr_uncentered=1.)),diagnostic=dict(source_video_loss=.5))
        failed=dict(result=dict(local_reference_metrics=dict(spider_sr_uncentered=0.)),diagnostic=dict(source_video_loss=.49))
        self.assertIs(quality.choose_candidate([failed,passed],.8),passed)
        passed['diagnostic']['source_video_loss']=.8
        self.assertIs(quality.choose_candidate([failed,passed],.8),failed)

    def test_resumed_maximum_budget_is_evaluated_and_advances_without_extending_again(self):
        with tempfile.TemporaryDirectory() as folder:
            base=Path(folder);out=base/'queue';out.mkdir();old=base/'old';old.mkdir();parent=base/'source';parent.mkdir()
            save(old/'episode-config.json',dict(fast_source_run=str(parent)))
            (parent/'reconstruction').mkdir();(parent/'reconstruction/scene_motion_physical.npz').write_bytes(b'source')
            save(out/'baseline.json',dict(episodes={'10':dict(run=str(old),trajectory='old-rollout')}))
            save(out/'quality-improvement-policy.json',dict(baseline_cohort=str(out/'baseline.json'),baseline_receipt='baseline-receipt'))
            root=base/'runs/episode_000010_newton_human_match_20261023_g001';root.mkdir(parents=True)
            save(root/'episode-config.json',dict(own_training_maximum_steps=2560000))
            save(root/'research/fast-submission-result.json',dict(actual_training_steps=2560000,checkpoint_sha256='candidate',trajectory='new-rollout',trajectory_sha256='rollout-hash'))
            child=Mock();child.poll.return_value=0;child.returncode=0
            with patch.object(quality,'BASE',base),patch.object(quality,'BATCH',base),\
                 patch.object(quality,'episode_slot',side_effect=lambda *a:contextlib.nullcontext()),\
                 patch('own_source_quality.prepare',return_value=parent),\
                 patch('source_policy_validation.score',return_value=dict(source_video_loss=1.)),\
                 patch('own_policy_calibration.diagnose',return_value=dict(dropped_frames=1)),\
                 patch('fast_policy_execution.select_result'),\
                 patch('own_submission_queue.eligible',return_value=dict(checkpoint_sha256='candidate',trajectory='new-rollout',trajectory_sha256='rollout-hash')),\
                 patch('own_submission_queue.run') as upload,\
                 patch('continuous_human_match_20261023.next_generation',return_value=1),\
                 patch('evaluation_experiment_queue.execution_inputs',return_value={}),\
                 patch('training_continuation.extend') as extend,\
                 patch('quality_optimizer_archive.archive'),\
                 patch('kaggle_quality_comparison.compare',return_value={}),\
                 patch.object(quality.subprocess,'check_output',return_value='commit'),\
                 patch.object(quality.subprocess,'Popen',return_value=child),\
                 patch.object(quality.time,'sleep',side_effect=AssertionError('Unexpected retry in recovery test')):
                quality.run(out,[10])
            extend.assert_not_called();upload.assert_called_once()
            progress=json.loads((out/'quality-training-status.json').read_text())
            self.assertEqual(progress['episodes']['10']['status'],'completed')
            self.assertFalse(progress['episodes']['10']['selected'])
            self.assertEqual(json.loads((out/'quality-selection.json').read_text())['episodes']['10']['run'],str(old))

    def test_better_sample_mean_does_not_mean_a_drop_free_rollout(self):
        result=dict(local_reference_metrics=dict(chord_sr=0.))
        diagnostic=dict(source_video_loss=.1,rollout_motion=dict(dropped_frames=12))
        self.assertFalse(quality.whole_rollout_passed(result,diagnostic))
        result['local_reference_metrics']['chord_sr']=1.
        self.assertFalse(quality.whole_rollout_passed(result,diagnostic))
        diagnostic['rollout_motion']['dropped_frames']=0
        self.assertTrue(quality.whole_rollout_passed(result,diagnostic))

    def test_successful_whole_checkpoint_is_retained_over_a_later_dropped_candidate(self):
        supported=dict(result=dict(checkpoint='whole-model-640k',local_reference_metrics=dict(chord_sr=1.)),
            diagnostic=dict(source_video_loss=.4,rollout_motion=dict(dropped_frames=0)))
        dropped=dict(result=dict(checkpoint='whole-model-1280k',local_reference_metrics=dict(chord_sr=0.)),
            diagnostic=dict(source_video_loss=.38,rollout_motion=dict(dropped_frames=12)))
        self.assertIs(quality.choose_candidate([supported,dropped],.8),supported)
