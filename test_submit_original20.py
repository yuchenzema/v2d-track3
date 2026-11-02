import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import submit_original20 as submit


class OriginalSubmissionTests(unittest.TestCase):
    def test_blocked_replay_cannot_upload_a_draft(self):
        with self.assertRaisesRegex(RuntimeError,'沒有上傳'):
            submit.ready_package({},dict(status='blocked',error='invalid reference',failure_log='native.log'))
        self.assertIsNone(submit.ready_package({},dict(status='running')))

    def test_complete_roster_and_verified_bytes_are_required(self):
        with tempfile.TemporaryDirectory() as directory:
            root=Path(directory);path=root/'track3_submission.parquet';path.write_bytes(b'validated bytes')
            request=dict(output=str(root),code_commit='producing-commit')
            status=dict(status='complete',episodes={str(ep):{} for ep in submit.EXPECTED},
                code_commit='producing-commit',package=dict(file=str(path),sha256=submit.sha(path),
                    official_packer_used=True,dry_run_passed=True))
            self.assertEqual(submit.ready_package(request,status),path)
            del status['episodes']['1']
            with self.assertRaisesRegex(AssertionError,'all 20'):submit.ready_package(request,status)
            status['episodes']['1']={};path.write_bytes(b'changed bytes')
            with self.assertRaisesRegex(AssertionError,'changed'):submit.ready_package(request,status)

    def test_existing_receipt_only_polls_without_upload_flags(self):
        with tempfile.TemporaryDirectory() as directory:
            root=Path(directory);queue=root/'queue';queue.mkdir()
            (queue/'submission-replay-request.json').write_text(json.dumps(dict(output=str(root))))
            (root/'original20.kaggle-receipt.json').write_text(json.dumps(dict(submissions={'auc':dict(ref=123)})))
            with patch.object(submit,'QUEUE',queue),patch('sys.argv',['submit_original20.py']),\
                 patch.object(submit.subprocess,'call',return_value=0) as call:
                self.assertEqual(submit.main(),0)
            command=call.call_args.args[0]
            self.assertIn('--poll-receipt',command)
            self.assertNotIn('--receipt',command)
            self.assertNotIn('-m',command)
