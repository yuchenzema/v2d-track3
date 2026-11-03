import json,tempfile,unittest
from pathlib import Path
from refresh_kaggle_status import reconcile_receipt


class ReceiptPollingTests(unittest.TestCase):
    def test_only_exact_competition_and_submission_id_can_complete_a_receipt(self):
        with tempfile.TemporaryDirectory() as directory:
            path=Path(directory)/'receipt.json'
            initial=dict(submissions=dict(add_auc=dict(ref=123,slug='auc',state='PENDING')))
            path.write_text(json.dumps(initial))
            data=dict(checked_utc='now',metrics=dict(add_auc=dict(slug='auc',submissions=[dict(ref=124,state='COMPLETE',public_score='.99')])))
            self.assertFalse(reconcile_receipt(path,data));self.assertEqual(json.loads(path.read_text()),initial)
            data['metrics']['add_auc']['submissions'][0]['ref']=123
            data['metrics']['add_auc']['slug']='another-competition'
            self.assertFalse(reconcile_receipt(path,data))
            data['metrics']['add_auc']['slug']='auc'
            self.assertTrue(reconcile_receipt(path,data))
            entry=json.loads(path.read_text())['submissions']['add_auc']
            self.assertEqual(entry['ref'],123);self.assertEqual(entry['state'],'COMPLETE');self.assertEqual(entry['public_score'],'.99')
