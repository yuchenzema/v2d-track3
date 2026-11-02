"""Exercise the real controller handoff without spawning reconstruction jobs."""
from contextlib import ExitStack
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import public_validation_pipeline as public
import source_cad_queue as cad
import continuous_human_match_20261023 as controller


class StopController(Exception):pass


class EvaluationAuditHandoffTests(unittest.TestCase):
    def setUp(self):
        self.temp=tempfile.TemporaryDirectory();self.addCleanup(self.temp.cleanup)
        self.out=Path(self.temp.name);self.val=self.out/'pipeline_validation_20261029';self.val.mkdir()
        self.public_episodes=[0,2];self.focus=[1,3,5,6,8,10,14,15,16,19,20,25,28,30,32,33,35,36,37,38]
        self.policy=dict(enabled=True,public_episodes=self.public_episodes,minimum_public_auc=.9,
                         camera_odometry_method='sift_pnp_metric',evaluation_reconstruction_after_public_coverage=True)
        self.results={}
        for ep in self.public_episodes:
            report=self.out/f'public-{ep}.json'
            self.write(report,dict(episode=ep,all_source_frames_checked=True,object_frames_checked=25,
                                   source_time_warping_used=False,physical_projection_applied=False))
            self.results[str(ep)]=dict(source_revision='',overlap_report=str(report),
                baseline=dict(kind='video_reconstructed_reference_not_policy',episode=ep,
                              evaluation_ground_truth_used=False,frames_scored=25),
                best_reference_metrics={'AUC':.7,'MP-SR':0.})
        self.write(self.out/'validated-pipeline-policy.json',self.policy)
        self.write(self.val/'status.json',dict(status='public_reference_validation_quality_insufficient',results=self.results))

    @staticmethod
    def write(path,data):path.write_text(json.dumps(data))

    def covered(self):return public.public_reconstruction_coverage_complete(self.results,self.public_episodes,{})

    def test_real_video_experiments_allow_audits_while_training_remains_blocked(self):
        self.assertTrue(self.covered())
        self.assertFalse(public.references_passed(self.results,self.public_episodes,.9,{}))

    def test_empty_or_missing_roster_never_admits_audits(self):
        self.assertFalse(public.public_reconstruction_coverage_complete({},[],{}))
        del self.results['2'];self.assertFalse(self.covered())

    def test_identity_control_is_not_a_video_experiment(self):
        self.results['2']['baseline']['kind']='public_identity_control'
        self.assertFalse(self.covered())

    def test_stale_source_revision_is_not_current_coverage(self):
        self.assertFalse(public.public_reconstruction_coverage_complete(self.results,self.public_episodes,{'2':'_new'}))

    def test_missing_partial_or_warped_source_audit_does_not_admit(self):
        report=Path(self.results['2']['overlap_report']);original=json.loads(report.read_text())
        for key,value in [('all_source_frames_checked',False),('object_frames_checked',0),
                          ('source_time_warping_used',True),('physical_projection_applied',True)]:
            with self.subTest(field=key):
                self.write(report,{**original,key:value});self.assertFalse(self.covered())
        report.unlink();self.assertFalse(self.covered())

    def test_public_worker_yields_to_audits_without_training_or_child_process(self):
        with ExitStack() as stack:
            for key,value in dict(OUT=self.out,VAL=self.val,ROOT=self.out,EPISODE=0).items():
                stack.enter_context(patch.object(public,key,value))
            child=stack.enter_context(patch.object(public.subprocess,'Popen'))
            setup=stack.enter_context(patch.object(public,'setup'))
            self.assertFalse(public.worker(self.out,self.focus))
            child.assert_not_called();setup.assert_not_called()
        state=json.loads((self.val/'status.json').read_text())
        self.assertTrue(state['evaluation_reconstruction_admitted'])
        self.assertFalse(state['evaluation_training_admitted']);self.assertEqual(state['training_steps'],0)

    def run_complete_audits(self,failed):
        self.write(self.out/'source-cad-status.json',dict(pipeline_version='fresh_stereo_sift_frontend_v4',
            results={str(ep):dict(local_overlap_failed_object_frames=failed,unobservable_object_frames=0) for ep in self.focus}))
        with patch.object(cad,'BATCH',self.out),patch.object(cad,'source_for') as source,patch.object(cad.subprocess,'Popen') as child:
            self.assertFalse(cad.run(self.out,self.focus,coverage_only=True))
            source.assert_not_called();child.assert_not_called()
        result=json.loads((self.out/'source-cad-status.json').read_text())
        self.assertEqual(result['inspected_count'],20);self.assertFalse(result['policy_training_allowed'])
        self.assertFalse(result['episode_done']);self.assertEqual(result['strict_completion_count'],0)
        self.assertEqual(json.loads((self.out/'status.json').read_text())['episodes_done'],[])

    def test_even_twenty_perfect_overlays_cannot_admit_training_in_audit_only_mode(self):self.run_complete_audits(0)

    def test_twenty_failed_overlays_finish_coverage_without_becoming_passed_episodes(self):self.run_complete_audits(4)

    def run_controller(self,expect_audits):
        with patch.object(controller,'OUT',self.out),patch.object(controller,'BATCH',self.out),\
             patch.object(controller,'FOCUS',self.focus),patch.object(public,'worker',return_value=False),\
             patch.object(cad,'run',side_effect=StopController) as audit,\
             patch.object(controller.time,'sleep',side_effect=StopController):
            with self.assertRaises(StopController):controller.main()
            if expect_audits:audit.assert_called_once_with(self.out,self.focus,coverage_only=True)
            else:audit.assert_not_called()

    def test_same_controller_routes_complete_public_coverage_to_evaluation_audits(self):self.run_controller(True)

    def test_incomplete_public_coverage_keeps_controller_on_public(self):
        del self.results['2']
        self.write(self.val/'status.json',dict(status='public_reference_validation_quality_insufficient',results=self.results))
        self.run_controller(False)

    def test_pending_reload_prevents_handoff_to_another_episode(self):
        self.write(self.out/'controller-reload-request.json',dict(pending=True));self.run_controller(False)
        with patch.object(cad,'BATCH',self.out),patch.object(cad,'source_for') as source:
            self.assertFalse(cad.run(self.out,self.focus,coverage_only=True));source.assert_not_called()
        state=json.loads((self.out/'source-cad-status.json').read_text())
        self.assertEqual(state['status'],'awaiting_safe_controller_reload');self.assertFalse(state['policy_training_allowed'])

    def test_failed_public_worker_cannot_trigger_audit_handoff(self):
        self.write(self.val/'status.json',dict(status='failed',results=self.results));self.run_controller(False)

    def test_handoff_is_opt_in(self):
        self.policy.pop('evaluation_reconstruction_after_public_coverage')
        self.write(self.out/'validated-pipeline-policy.json',self.policy);self.run_controller(False)


if __name__=='__main__':unittest.main()
