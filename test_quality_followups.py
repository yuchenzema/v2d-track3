import json,tempfile,unittest
from pathlib import Path
from unittest.mock import patch
from improvement_common import save
from quality_followups import admit,improving,apply_repair_requests


class MeasuredFollowupTests(unittest.TestCase):
    def test_measured_repair_starts_one_fresh_job_and_keeps_the_previous_run(self):
        with tempfile.TemporaryDirectory() as directory:
            out=Path(directory);root=out/'original';save(root/'episode-config.json',dict(own_geometry_required=True))
            progress=dict(episodes={'16':dict(status='completed',run=str(root),request='original-job')})
            save(out/'quality-repair-requests.json',dict(episodes={'16':dict(source_run=str(root),basis='Verified static wrench fit; unchanged frozen reference')}))
            with patch('own_geometry.validate'):
                apply_repair_requests(out,progress);apply_repair_requests(out,progress)
            self.assertEqual(progress['episodes']['16']['status'],'contact_refit_pending')
            self.assertNotIn('request',progress['episodes']['16']);self.assertEqual(len(progress['followup_history']),1)
            self.assertEqual(progress['followup_history'][0]['previous']['request'],'original-job')

    def test_more_training_requires_measured_progress_without_extra_drops(self):
        old=dict(actual_training_steps=1280000,source_video_loss=1.,rollout_motion=dict(dropped_frames=10))
        new=dict(actual_training_steps=2560000,source_video_loss=.9,rollout_motion=dict(dropped_frames=9))
        self.assertTrue(improving([new,old]))
        new['rollout_motion']['dropped_frames']=11;self.assertFalse(improving([old,new]))
        new['rollout_motion']['dropped_frames']=9;new['source_video_loss']=.98;self.assertFalse(improving([old,new]))

    def test_legacy_reference_is_rebuilt_in_a_fresh_job_and_successful_models_are_retained(self):
        with tempfile.TemporaryDirectory() as directory:
            out=Path(directory);progress=dict(episodes={})
            for ep,success in [(35,0.),(37,1.)]:
                root=out/f'run{ep}';save(root/'episode-config.json',dict(own_geometry_required=True,fast_source_run=f'/own/evaluation_episode_{ep:06d}_quality_20261103_v1'))
                save(root/'research/fast-submission-result.json',dict(local_reference_metrics=dict(chord_sr=success)))
                progress['episodes'][str(ep)]=dict(status='completed',run=str(root),request='original-job')
            with patch('own_geometry.validate'),patch('training_continuation.extend') as extend:
                self.assertEqual(admit(out,[35,37],progress,{}),[35])
            extend.assert_not_called()
            self.assertEqual(progress['episodes']['35']['status'],'source_refit_pending')
            self.assertNotIn('request',progress['episodes']['35'])
            self.assertEqual(progress['episodes']['37']['status'],'completed')
            self.assertEqual(progress['followup_history'][0]['previous']['request'],'original-job')
