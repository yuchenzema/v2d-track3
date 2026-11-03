import contextlib,json,tempfile,unittest
from pathlib import Path
from unittest.mock import patch,Mock
import own_policy_quality_training as quality
from improvement_common import save


class QualityRecoveryTests(unittest.TestCase):
    def test_resumed_maximum_budget_is_evaluated_and_advances_without_extending_again(self):
        with tempfile.TemporaryDirectory() as folder:
            base=Path(folder);out=base/'queue';out.mkdir();old=base/'old';old.mkdir();parent=base/'source';parent.mkdir()
            save(old/'episode-config.json',dict(fast_source_run=str(parent)))
            (parent/'reconstruction').mkdir();(parent/'reconstruction/scene_motion_physical.npz').write_bytes(b'source')
            save(out/'baseline.json',dict(episodes={'10':dict(run=str(old),trajectory='old-rollout')}))
            save(out/'quality-improvement-policy.json',dict(baseline_cohort=str(out/'baseline.json'),baseline_receipt='baseline-receipt'))
            root=base/'runs/episode_000010_newton_human_match_20261023_g001';root.mkdir(parents=True)
            save(root/'episode-config.json',dict(own_training_maximum_steps=2560000))
            save(root/'research/fast-submission-result.json',dict(actual_training_steps=2560000,checkpoint_sha256='candidate',trajectory='new-rollout'))
            child=Mock();child.poll.return_value=0;child.returncode=0
            with patch.object(quality,'BASE',base),patch.object(quality,'BATCH',base),\
                 patch.object(quality,'episode_slot',side_effect=lambda *a:contextlib.nullcontext()),\
                 patch('own_source_quality.prepare',return_value=parent),\
                 patch('source_policy_validation.score',return_value=dict(source_video_loss=1.)),\
                 patch('own_submission_queue.eligible',return_value=dict(checkpoint_sha256='candidate',trajectory='new-rollout')),\
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
