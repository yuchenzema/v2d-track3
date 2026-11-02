from contextlib import contextmanager
import json
from pathlib import Path
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch

import fresh_evaluation_frontend as frontend
import source_cad_queue as queue


class SourceCadQueueStateTests(unittest.TestCase):
    def test_new_experimental_candidate_yields_at_serial_boundary_without_starting_another_cad(self):
        import evaluation_experiment_queue as experiments
        with tempfile.TemporaryDirectory() as directory:
            out=Path(directory)
            (out/'experimental-training-policy.json').write_text('{"enabled":true}')
            with patch.object(queue,'BATCH',out),patch.object(queue,'source_for') as source,\
                 patch.object(queue.subprocess,'Popen') as child,\
                 patch.object(experiments,'candidates',return_value=[(38,{}, {})]) as eligible:
                self.assertFalse(queue.run(out,[38]))
            source.assert_not_called();child.assert_not_called();eligible.assert_called_once()
            state=json.loads((out/'source-cad-status.json').read_text())
            self.assertEqual(state['status'],'experimental_candidate_ready')
            self.assertFalse(state['policy_training_allowed'])
            self.assertEqual(state['strict_completion_count'],0)
    def exercise(self,fail_cad=False):
        with tempfile.TemporaryDirectory() as directory:
            base=Path(directory);(base/'runs').mkdir();out=base/'queue';out.mkdir()
            policy=dict(enabled=True,camera_odometry_method='sift_pnp_metric')
            (out/'validated-pipeline-policy.json').write_text(json.dumps(policy))
            (out/'source-cad-status.json').write_text(json.dumps(dict(pipeline_version='fresh_stereo_sift_frontend_v4',
                results={},attempts={'3':2},latest_failure='Earlier missing source depth frame')))
            root=base/'runs/episode_000003_cad_source_validated_v4_20261029'
            (root/'research').mkdir(parents=True);overlay=root/'research/overlay.mp4';overlay.write_bytes(b'overlay control')
            (root/'research/cad-alignment-report.json').write_text(json.dumps(dict(
                all_source_frames_checked=True,overlay=str(overlay),local_overlap_failed_object_frames=4,unobservable_object_frames=0)))
            held=False;seen=[]
            @contextmanager
            def slot(run,episode):
                nonlocal held
                self.assertFalse(held);held=True
                try:yield
                finally:held=False
            def frontend_run(source,episode,progress):
                self.assertTrue(held);progress(base/'frontend','generic');return base/'frontend'
            def launch(*args,**kwargs):
                self.assertTrue(held)
                state=json.loads((out/'status.json').read_text());cad=json.loads((out/'source-cad-status.json').read_text())
                self.assertEqual(state['status'],'source_cad_alignment')
                self.assertEqual(state['active_optimization']['run'],str(root))
                self.assertEqual(state['active_optimization']['dataset_split'],'evaluation')
                self.assertEqual(cad['stage'],'CAD_geometry_and_source_pose_fit');self.assertNotIn('latest_failure',cad)
                self.assertEqual(cad['previous_failure']['message'],'Earlier missing source depth frame');seen.append(3)
                return SimpleNamespace(pid=123,poll=lambda:0,returncode=1 if fail_cad else 0)
            def stop_after_failure(seconds):raise InterruptedError('Stop control after recorded failure')
            with patch.object(queue,'BASE',base),patch.object(queue,'BATCH',base),\
                 patch.object(queue,'source_for',return_value=base/'source'),\
                 patch.object(queue,'episode_slot',side_effect=slot),\
                 patch.object(queue,'register_episode_child'),patch.object(queue.subprocess,'Popen',side_effect=launch),\
                 patch.object(frontend,'run',side_effect=frontend_run),\
                 patch.object(queue.time,'sleep',side_effect=stop_after_failure):
                if fail_cad:
                    with self.assertRaises(InterruptedError):queue.run(out,[3],coverage_only=True)
                else:self.assertFalse(queue.run(out,[3],coverage_only=True))
            self.assertEqual(seen,[3]);self.assertFalse(held)
            cad=json.loads((out/'source-cad-status.json').read_text());state=json.loads((out/'status.json').read_text())
            self.assertFalse(cad['policy_training_allowed']);self.assertEqual(cad['strict_completion_count'],0)
            self.assertEqual(state['episodes_done'],[])
            return cad

    def test_frontend_to_cad_handoff_reports_actual_run_and_archives_old_failure(self):
        cad=self.exercise();self.assertNotIn('latest_failure',cad);self.assertEqual(cad['inspected_count'],1)

    def test_new_cad_failure_is_reported_as_current_without_erasing_history(self):
        cad=self.exercise(fail_cad=True)
        self.assertEqual(cad['status'],'source_alignment_failed')
        self.assertIn('CAD source fit exited 1',cad['latest_failure'])
        self.assertEqual(cad['previous_failure']['episode'],3)


if __name__=='__main__':unittest.main()
