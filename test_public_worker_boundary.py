from contextlib import contextmanager,ExitStack
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import public_validation_pipeline as pipeline


class PublicWorkerBoundaryTests(unittest.TestCase):
    def run_worker(self,pending,already_passed=False,previous_failure=False):
        with tempfile.TemporaryDirectory() as directory:
            base=Path(directory);out=base/'queue';out.mkdir();val=out/'validation';val.mkdir()
            (out/'validated-pipeline-policy.json').write_text(json.dumps(dict(public_episodes=[0,2],minimum_public_auc=.9)))
            (out/'controller-reload-request.json').write_text(json.dumps(dict(pending=pending)))
            good=dict(AUC=.95,**{'SP-SR':1.,'MP-SR':1.},RPE_cm=1.,MPPE_cm=1.)
            configured=[];held=False;entered=[];released=[]
            def configure(episode):
                pipeline.EPISODE=episode;pipeline.ROOT=base/f'input_{episode}';configured.append(episode)
            def setup():
                root=pipeline.ROOT;(root/'reconstruction').mkdir(parents=True,exist_ok=True);(root/'logs').mkdir(exist_ok=True)
                for marker in ['video-preparation.json','sam2_prompts.json','stereo-depth-alignment.json','masks-reviewed.json','scene_motion_physical.npz']:
                    (root/'reconstruction'/marker).write_bytes(b'unit control')
                episode=pipeline.EPISODE
                baseline={**good,'AUC':.8,'RPE_cm':2.,'MPPE_cm':2.}
                (val/f'public_{episode:06d}_reference_baseline.json').write_text(json.dumps(dict(metrics=baseline)))
                aligned=base/f'runs/public_episode_{episode:06d}_cad_alignment_20261029'
                (aligned/'research').mkdir(parents=True,exist_ok=True);(aligned/'reconstruction').mkdir(exist_ok=True)
                (aligned/'research/cad-alignment-report.json').write_text('{}')
                (aligned/'reconstruction/cad-source-trajectory.npz').write_bytes(b'digest control; not a rollout')
                for advance in [-2.,-1.25,0.,1.25,2.]:
                    (val/f'public_{episode:06d}_advance_{advance:+.2f}.json').write_text(json.dumps(dict(metrics=good)))
                return {}
            @contextmanager
            def slot(root,episode):
                nonlocal held
                self.assertFalse(held);held=True;entered.append(episode)
                try:yield
                finally:held=False;released.append(episode)
            original_save=pipeline.save
            def save(path,data):
                if data.get('status')=='awaiting_safe_controller_reload':
                    self.assertFalse(held,'Reload was requested before the protected episode was released')
                original_save(path,data)
            initial={}
            if already_passed:initial['results']={str(ep):dict(refined_metrics=good,best_reference_metrics=good) for ep in [0,2]}
            if previous_failure:initial.update(status='failed',error='Earlier source-mask propagation failed',error_report='earlier-source/logs/failure.txt')
            if initial:(val/'status.json').write_text(json.dumps(initial))
            with ExitStack() as stack:
                for key,value in dict(BASE=base,OUT=out,VAL=val,ROOT=base/'input_0',EPISODE=0,SOURCE_REVISION='').items():
                    stack.enter_context(patch.object(pipeline,key,value))
                stack.enter_context(patch.object(pipeline,'configure',side_effect=configure))
                stack.enter_context(patch.object(pipeline,'setup',side_effect=setup))
                stack.enter_context(patch.object(pipeline,'episode_slot',side_effect=slot))
                stack.enter_context(patch.object(pipeline,'save',side_effect=save))
                popen=stack.enter_context(patch.object(pipeline.subprocess,'Popen',side_effect=AssertionError('Control fixture must not launch a worker')))
                result=pipeline.worker(out,[19,20]);popen.assert_not_called()
            return result,json.loads((val/'status.json').read_text()),configured,entered,released

    def test_pending_reload_yields_after_completed_episode_and_before_next(self):
        result,status,configured,entered,released=self.run_worker(True)
        self.assertFalse(result);self.assertEqual(configured,[0]);self.assertEqual(entered,[0]);self.assertEqual(released,[0])
        self.assertEqual(set(status['results']),{'0'});self.assertEqual(status['status'],'awaiting_safe_controller_reload')
        self.assertIsNone(status['active_episode']);self.assertFalse(status['evaluation_training_admitted'])

    def test_no_pending_reload_keeps_the_serial_roster_running(self):
        result,status,configured,entered,released=self.run_worker(False)
        self.assertTrue(result);self.assertEqual(configured,[0,2]);self.assertEqual(entered,[0,2]);self.assertEqual(released,[0,2])
        self.assertEqual(set(status['results']),{'0','2'})

    def test_pending_reload_blocks_admission_even_when_all_references_were_already_passed(self):
        result,status,_,entered,_=self.run_worker(True,already_passed=True)
        self.assertFalse(result);self.assertEqual(entered,[])
        self.assertFalse(status['evaluation_training_admitted'])

    def test_successful_resume_keeps_old_failure_as_history_not_current_blocker(self):
        result,status,_,entered,_=self.run_worker(False,previous_failure=True)
        self.assertTrue(result);self.assertEqual(entered,[0,2])
        self.assertNotIn('error',status);self.assertNotIn('error_report',status)
        self.assertEqual(status['previous_validation_failure']['error'],'Earlier source-mask propagation failed')
        self.assertEqual(status['previous_validation_failure']['error_report'],'earlier-source/logs/failure.txt')
