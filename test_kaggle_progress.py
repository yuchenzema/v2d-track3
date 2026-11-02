import json
from pathlib import Path
import tempfile
import unittest
from kaggle_progress import receipt_scores,METRICS


class KaggleProgressTests(unittest.TestCase):
    def test_scored_submission_is_separate_from_strict_episode_completion(self):
        with tempfile.TemporaryDirectory() as d:
            path=Path(d)/'receipt.json'
            path.write_text(json.dumps(dict(submissions={k:dict(state='COMPLETE',public_score='0.0',ref=i+1,slug='metric') for i,(k,*_) in enumerate(METRICS)})))
            scores=receipt_scores(path)
            self.assertEqual(scores['completed'],6)
            self.assertFalse(scores['counts_as_strict_episode_completion'])
            self.assertEqual(next(r for r in scores['metrics'] if r['name']=='CD-O')['direction'],'lower')
            self.assertEqual(next(r for r in scores['metrics'] if r['name']=='AUC')['direction'],'higher')

    def test_invalid_or_pending_scores_do_not_count_as_complete(self):
        with tempfile.TemporaryDirectory() as d:
            path=Path(d)/'receipt.json'
            path.write_text(json.dumps(dict(submissions={'add_auc':dict(state='COMPLETE',public_score='nan')})))
            scores=receipt_scores(path);self.assertEqual(scores['completed'],0)
            self.assertIsNone(scores['metrics'][0]['score'])
