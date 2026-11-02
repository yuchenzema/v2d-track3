import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch
import full_roster_readiness as review

class NewPipelineTests(unittest.TestCase):
    def setUp(self):
        self.temp=tempfile.TemporaryDirectory();self.addCleanup(self.temp.cleanup);self.root=Path(self.temp.name)
        self.output=self.root/'output';self.output.mkdir();(self.root/'research').mkdir();(self.root/'deliverables').mkdir();(self.root/'flash_chord_coherent_actor_transfer').mkdir()
        self.request=self.root/'request.json';self.request.write_text(json.dumps({'started_utc':'2026-10-27T05:00:00Z'}))
        (self.root/'episode-config.json').write_text(json.dumps({'frames':202,'match_request':str(self.request),'match_output':str(self.output)}))
        (self.root/'research/full-reconstruction-completed.json').touch();(self.root/'flash_chord_coherent_actor_transfer/TRAINING_COMPLETED').touch()
        (self.output/'episode_000020-fresh-result.json').write_text(json.dumps({'status':'completed','artifact_integrity':{'integrity_passed':True}}))
        self.manifest={name:{'sha256':name} for name in ['original_vs_policy.mp4','rollout_source_timed.mp4','report.json','policy_states.npz']}
        (self.root/'deliverables/manifest.json').write_text(json.dumps(self.manifest))
    def evaluate(self):return review.attempt_review(self.root,20,'2026-10-27T04:00:00Z',set())
    def write_audit(self,version):
        (self.root/'research/strict-human-match.json').write_text(json.dumps({'version':version,'states_sha256':'policy_states.npz','all_frames_checked':True,'source_frames':202,'passes_strict_local_checks':True}))
    def test_training_and_export_without_pinned_audit_are_incomplete(self):
        self.assertFalse(self.evaluate()['pipeline_complete'])
    def test_stale_audit_cannot_qualify_a_new_rollout(self):
        self.write_audit('old-version');self.assertFalse(self.evaluate()['pipeline_complete'])
        self.write_audit(review.manifest_version(self.manifest));self.assertTrue(self.evaluate()['pipeline_complete'])
    def test_historical_completed_run_does_not_count_for_new_round(self):
        self.request.write_text(json.dumps({'started_utc':'2026-10-20T00:00:00Z'}))
        self.write_audit(review.manifest_version(self.manifest));self.assertFalse(self.evaluate()['pipeline_complete'])
        self.assertTrue(review.attempt_review(self.root,20,'2026-10-27T04:00:00Z',{str(self.root)})['pipeline_complete'])

class SubmissionTests(unittest.TestCase):
    def test_packaging_alone_does_not_override_failed_motion(self):
        attempt={'pipeline_complete':True,'motion_passed':False}
        rows=[{'episode':1,'videos':{'latest':{'version':'v','run':'r'}}}]
        policy={'episodes':[1],'round_epoch':'now'}
        loop={'episodes':{'1':{'version':'v','passes_strict_local_checks':False}},'current_round_episodes_checked':1,'current_round_source_frames_checked':335}
        with patch.object(review,'attempt_review',return_value=attempt),patch.object(review,'read',return_value={}):
            result=review.build(rows,loop,policy,{'final_submission_package_ready':True})
        self.assertEqual(result['new_pipelines_completed'],1);self.assertFalse(result['ready_for_submission'])

if __name__=='__main__':unittest.main()
