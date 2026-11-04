import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch
import own_submission_queue as queue


class OwnSubmissionTests(unittest.TestCase):
    def test_measured_whole_episode_override_applies_only_to_the_known_regressing_run(self):
        with tempfile.TemporaryDirectory() as directory:
            out=Path(directory)
            (out/'quality-selection.json').write_text(json.dumps(dict(episodes={'32':dict(run='known-regression')})))
            (out/'quality-kaggle-selection-overrides.json').write_text(json.dumps(dict(episodes={'32':dict(rejected_run='known-regression',selection=dict(run='measured-better'))})))
            self.assertEqual(queue.selections(out)['32']['run'],'measured-better')
            (out/'quality-selection.json').write_text(json.dumps(dict(episodes={'32':dict(run='fresh-geometry')})))
            self.assertEqual(queue.selections(out)['32']['run'],'fresh-geometry')

    def test_existing_full_cohort_trial_ids_are_reused_without_uploading_again(self):
        with tempfile.TemporaryDirectory() as directory:
            batch=Path(directory);out=batch/'focused_episodes_20261028';out.mkdir()
            (out/'own-geometry-policy.json').write_text(json.dumps(dict(enabled=True,automatic_upload_after_full_verification=True)))
            ready={str(ep):dict(run=f'run{ep}') for ep in range(20)}
            trial=batch/'trials/complete';trial.mkdir(parents=True);receipt=trial/'kaggle-receipt.json'
            receipt.write_text(json.dumps(dict(submissions={m:dict(state='COMPLETE',ref=i+1) for i,m in enumerate(['add_auc','spider_sr','maniptrans_sr','rpe_cm','mppe_cm','cd_o_cm'])})))
            (trial/'cohort-provenance.json').write_text(json.dumps(dict(episodes=ready)))
            (out/'own-submission-status.json').write_text(json.dumps(dict(receipt=str(receipt))))
            with patch.object(queue,'BATCH',batch),patch.object(queue,'collect',return_value=(ready,{})),patch.object(queue.subprocess,'run') as upload:
                self.assertFalse(queue.run(out,list(range(20))))
            upload.assert_not_called()
            self.assertEqual(json.loads((out/'own-submission-status.json').read_text())['receipt'],str(receipt))

    def test_failed_upload_without_id_never_waits_for_scoring(self):
        result=queue.receipt_states(dict(submissions=dict(
            spider_sr=dict(state='UPLOAD_FAILED',ref=None),
            add_auc=dict(state='PENDING',ref=123),
            cd_o_cm=dict(state='COMPLETE',ref=124))))
        self.assertEqual(result,dict(failed=['spider_sr'],pending=['add_auc']))

    def test_recovery_keeps_other_metrics_and_never_replaces_an_accepted_id(self):
        with tempfile.TemporaryDirectory() as directory:
            out=Path(directory);parent=out/'parent.json';recovery=out/'retry.json'
            original=dict(track='track_3',file=str(out/'cohort.parquet'),submissions=dict(
                add_auc=dict(slug='auc',ref=11,state='COMPLETE',public_score='.2'),
                spider_sr=dict(slug='sp',ref=None,state='UPLOAD_FAILED')))
            retry=dict(track='track_3',file=original['file'],submissions=dict(spider_sr=dict(slug='sp',ref=12,state='PENDING')))
            parent.write_text(json.dumps(original));recovery.write_text(json.dumps(retry))
            self.assertEqual(queue.attach_failed_upload_recovery(parent,recovery),['spider_sr'])
            updated=json.loads(parent.read_text());self.assertEqual(updated['submissions']['add_auc'],original['submissions']['add_auc'])
            self.assertEqual(queue.attach_failed_upload_recovery(parent,recovery),[])
            retry['submissions']['spider_sr']['ref']=13;recovery.write_text(json.dumps(retry))
            with self.assertRaisesRegex(AssertionError,'Existing accepted IDs'):queue.attach_failed_upload_recovery(parent,recovery)
            self.assertEqual(json.loads(parent.read_text())['submissions']['spider_sr']['ref'],12)

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
