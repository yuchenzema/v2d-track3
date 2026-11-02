import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch
import own_submission_queue as queue


class OwnSubmissionTests(unittest.TestCase):
    def test_incomplete_twenty_episode_cohort_never_uploads(self):
        with tempfile.TemporaryDirectory() as directory:
            out=Path(directory);(out/'own-geometry-policy.json').write_text(json.dumps(dict(enabled=True,automatic_upload_after_full_verification=True)))
            with patch.object(queue,'collect',return_value=({'1':{}},{'3':['not trained']})),patch.object(queue.subprocess,'run') as upload:
                self.assertFalse(queue.run(out,list(range(20))))
            upload.assert_not_called()
            self.assertFalse(json.loads((out/'own-submission-status.json').read_text())['strict_completion_granted'])

    def test_historical_actor_or_scan_cannot_be_submitted(self):
        with tempfile.TemporaryDirectory() as directory:
            root=Path(directory);config=root/'episode-config.json';config.write_text('{}')
            with self.assertRaisesRegex(ValueError,'scan-derived'):queue.eligible(root)
            config.write_text(json.dumps(dict(own_geometry_required=True,actor_transfer={'checkpoint':'old'})))
            with self.assertRaisesRegex(ValueError,'actor lineage'):queue.eligible(root)
