import json
from pathlib import Path
import tempfile
import unittest
from kaggle_progress import receipt_scores,compare_receipts,current_scores,METRICS


class KaggleProgressTests(unittest.TestCase):
    def test_pending_candidate_keeps_one_complete_cohort_visible_without_mixing_scores(self):
        with tempfile.TemporaryDirectory() as directory:
            batch=Path(directory);old=batch/'own_geometry_submissions/old/kaggle-receipt.json';new=batch/'own_geometry_submissions/new/kaggle-receipt.json'
            old.parent.mkdir(parents=True);new.parent.mkdir(parents=True)
            old.write_text(json.dumps(dict(submissions={k:dict(state='COMPLETE',public_score='0.25',ref=i+1) for i,(k,*_) in enumerate(METRICS)})))
            new.write_text(json.dumps(dict(submissions={k:dict(state='COMPLETE' if i==0 else 'PENDING',public_score='0.99' if i==0 else None,ref=i+11) for i,(k,*_) in enumerate(METRICS)})))
            out=batch/'focused_episodes_20261028';out.mkdir();(out/'own-submission-status.json').write_text(json.dumps(dict(receipt=str(new))))
            scores=current_scores(batch)
            self.assertEqual(scores['receipt'],str(old));self.assertEqual(scores['completed'],6)
            self.assertEqual({row['score'] for row in scores['metrics']},{.25})
            self.assertEqual(scores['pending_candidate']['receipt'],str(new));self.assertEqual(scores['pending_candidate']['completed'],1)

    def test_live_administrative_invalidation_overrides_cached_complete_scores(self):
        with tempfile.TemporaryDirectory() as directory:
            root=Path(directory);path=root/'receipt.json'
            path.write_text(json.dumps(dict(submissions={'add_auc':dict(state='COMPLETE',public_score='.99',ref=123)})))
            (root/'kaggle-live-status.json').write_text(json.dumps(dict(checked_utc='2026-11-02T19:00:00Z',metrics={
                'add_auc':dict(submissions=[dict(ref=123,state='ERROR',public_score=None,error='Submission has been administratively invalidated')])})))
            scores=receipt_scores(path)
            self.assertEqual(scores['completed'],0);self.assertEqual(scores['invalidated'],1)
            self.assertIsNone(scores['metrics'][0]['score']);self.assertTrue(scores['metrics'][0]['live_verified'])
            comparison=compare_receipts(path,path)
            self.assertEqual(comparison['improved'],0);self.assertFalse(comparison['non_regressing'])
    def test_partial_or_mixed_improvements_cannot_claim_all_six_improved(self):
        with tempfile.TemporaryDirectory() as d:
            old=Path(d)/'old.json';new=Path(d)/'new.json'
            baseline=dict(submissions={k:dict(state='COMPLETE',public_score='1.00000') for k,*_ in METRICS})
            candidate=json.loads(json.dumps(baseline));candidate['submissions']['add_auc']['public_score']='1.1';candidate['submissions']['cd_o_cm']['public_score']='1.01'
            old.write_text(json.dumps(baseline));new.write_text(json.dumps(candidate))
            comparison=compare_receipts(old,new)
            self.assertEqual(comparison['improved'],1);self.assertEqual(comparison['regressed'],1)
            self.assertFalse(comparison['all_six_strictly_improved']);self.assertFalse(comparison['non_regressing'])
            candidate['submissions']['cd_o_cm']['public_score']='0.99';new.write_text(json.dumps(candidate))
            self.assertTrue(compare_receipts(old,new)['non_regressing'])
            self.assertFalse(compare_receipts(old,new)['all_six_strictly_improved'])

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
