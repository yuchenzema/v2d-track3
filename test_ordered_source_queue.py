import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch
import sequential_30_then_20_20261026 as queue
import video_catalog as catalog

class OrderedQueueTests(unittest.TestCase):
    def test_training_alone_does_not_release_successor(self):
        with tempfile.TemporaryDirectory() as d,patch.object(queue,'PREDECESSOR',Path(d)):
            (Path(d)/'status.json').write_text(json.dumps({'status':'completed'}))
            self.assertFalse(queue.predecessor_finished())

    def test_requires_export_integrity_and_independent_audit(self):
        with tempfile.TemporaryDirectory() as d,patch.object(queue,'PREDECESSOR',Path(d)):
            p=Path(d);(p/'human_review/episode_000030').mkdir(parents=True)
            (p/'status.json').write_text(json.dumps({'status':'completed'}))
            (p/'episode_000030-fresh-result.json').write_text(json.dumps({'status':'completed','artifact_integrity':{'integrity_passed':True}}))
            (p/'human_review/episode_000030/report.json').write_text(json.dumps({'independent_fresh_training_completed':True}))
            self.assertTrue(queue.predecessor_finished())

class StrictPublicationTests(unittest.TestCase):
    def test_missing_or_stale_audit_cannot_publish(self):
        with tempfile.TemporaryDirectory() as d:
            p=Path(d);(p/'research').mkdir()
            self.assertFalse(catalog.strict_release_matches(p,'new',{}))
            (p/'research/strict-human-match.json').write_text(json.dumps({'passes_strict_local_checks':True,'version':'old','states_sha256':'hash'}))
            self.assertFalse(catalog.strict_release_matches(p,'new',{}))

    def test_matching_audit_requires_exact_recorded_state_hash(self):
        with tempfile.TemporaryDirectory() as d:
            p=Path(d);(p/'research').mkdir()
            (p/'research/strict-human-match.json').write_text(json.dumps({'passes_strict_local_checks':True,'version':'new','states_sha256':'hash'}))
            with patch.object(catalog,'verified',side_effect=ValueError('mismatch')):
                self.assertFalse(catalog.strict_release_matches(p,'new',{'recorded_states':'states.npz'}))
            with patch.object(catalog,'verified') as check:
                from test_rollout_collision_validation import passing_collision
                collision=passing_collision();collision.update(states_sha256='hash',version='new')
                (p/'research/rollout-collision-validation.json').write_text(json.dumps(collision))
                self.assertTrue(catalog.strict_release_matches(p,'new',{'recorded_states':'states.npz'}))
                check.assert_called_once_with(p/'states.npz','hash')

if __name__=='__main__':unittest.main()
