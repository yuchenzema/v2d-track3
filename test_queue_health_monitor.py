import tempfile
from pathlib import Path
import unittest
import json
import fcntl
import signal
import threading
import contextlib
import io
from unittest.mock import patch
import queue_health_monitor as monitor

from queue_health_monitor import age, latest_metric, recovery_decision


class QueueHealthTests(unittest.TestCase):
    def test_exhausted_idle_alignment_is_not_healthy_progress(self):
        state=dict(status='source_alignment_repairs_required')
        self.assertIn('No episode worker',monitor.source_alignment_idle_problem(state,False,False))
        self.assertIsNone(monitor.source_alignment_idle_problem(state,True,False))
        self.assertIsNone(monitor.source_alignment_idle_problem(state,False,True))
        self.assertIsNone(monitor.source_alignment_idle_problem(dict(status='experimental_policy_training'),False,False))

    def test_frontend_revision_is_applied_at_the_locked_boundary_before_restart(self):
        with tempfile.TemporaryDirectory() as directory:
            root=Path(directory);(root/'controller-reload-request.json').write_text(json.dumps(dict(pending=True,frontend_revisions={'24':'_review4'})))
            (root/'frontend-revisions.json').write_text('{"23":"_boards2"}')
            def restart(*args,**kwargs):
                self.assertEqual(json.loads((root/'frontend-revisions.json').read_text()),{'23':'_boards2','24':'_review4'})
            with patch.object(monitor,'OUT',root),patch.object(monitor,'BATCH',root),patch.object(monitor,'LOCK',root/'lock'),patch.object(monitor,'owner_alive',return_value=False),patch.object(monitor,'native_learners',return_value=[]),patch.object(monitor.subprocess,'run',side_effect=restart) as run:
                self.assertTrue(monitor.apply_pending_update(False,False));run.assert_called_once()
            self.assertEqual(json.loads((root/'controller-reload-request.json').read_text())['applied_frontend_revisions'],{'24':'_review4'})

    def test_frontend_revision_cannot_escape_its_episode_directory(self):
        with tempfile.TemporaryDirectory() as directory:
            root=Path(directory);(root/'controller-reload-request.json').write_text(json.dumps(dict(pending=True,frontend_revisions={'24':'../../other'})))
            with patch.object(monitor,'OUT',root),patch.object(monitor,'BATCH',root),patch.object(monitor,'LOCK',root/'lock'),patch.object(monitor,'owner_alive',return_value=False),patch.object(monitor,'native_learners',return_value=[]),patch.object(monitor.subprocess,'run') as run:
                with self.assertRaisesRegex(ValueError,'Unsafe'):monitor.apply_pending_update(False,False)
                run.assert_not_called()
            self.assertFalse((root/'frontend-revisions.json').exists())

    def test_public_native_failure_is_not_hidden_by_an_active_controller(self):
        status,problem=monitor.public_validation_health(dict(status='failed',active_episode=24,error='trace\nBoard detection failed'),False)
        self.assertEqual(status,'public_validation_failed')
        self.assertIn('24',problem);self.assertIn('Board detection failed',problem)

    def test_active_public_repair_does_not_report_the_previous_failure_as_current(self):
        status,problem=monitor.public_validation_health(dict(status='failed',active_episode=24),True)
        self.assertEqual(status,'public_pipeline_validation');self.assertIsNone(problem)

    def test_exhausted_public_quality_is_visible_as_blocked_without_relaxing_admission(self):
        status,problem=monitor.public_validation_health(dict(status='public_reference_validation_quality_insufficient'),False)
        self.assertEqual(status,'public_reference_quality_blocked');self.assertIn('training is blocked',problem)
        self.assertEqual(monitor.public_validation_health(dict(status='awaiting_safe_controller_reload'),False),('public_controller_reload_pending',None))

    def test_user_pause_takes_priority_over_public_failure(self):
        self.assertEqual(monitor.public_validation_health(dict(status='failed'),False,True),('paused_by_user',None))

    def test_native_public_failure_reaches_saved_health_report_without_spawning_work(self):
        with tempfile.TemporaryDirectory() as directory:
            root=Path(directory);out=root/'queue';out.mkdir();(out/'pipeline_validation_20261029').mkdir()
            (out/'status.json').write_text(json.dumps(dict(status='public_pipeline_validation',execution_plan={'19':dict(automatic=True)})))
            (out/'pipeline_validation_20261029/status.json').write_text(json.dumps(dict(status='failed',active_episode=24,error='Source prompt failure')))
            with patch.object(monitor,'OUT',out),patch.object(monitor,'BATCH',root),patch.object(monitor,'EXPECTED',{19}),patch.object(monitor,'service_state',return_value='active'),patch.object(monitor,'owner_alive',return_value=False),patch.object(monitor,'native_learners',return_value=[]),patch.object(monitor,'available_ram_bytes',return_value=64*1024**3),patch.object(monitor.shutil,'disk_usage',return_value=type('Disk',(),dict(free=100_000_000_000))()),patch.object(monitor.subprocess,'run') as run,contextlib.redirect_stdout(io.StringIO()):
                monitor.main();run.assert_not_called()
            report=json.loads((out/'health-monitor.json').read_text())
            self.assertEqual(report['status'],'needs_attention')
            self.assertEqual(report['task_status'],'public_validation_failed')
            self.assertIn('Source prompt failure',report['public_blocking_failure'])

    def test_reload_lock_waits_for_a_short_episode_boundary(self):
        with tempfile.TemporaryDirectory() as directory:
            path=Path(directory)/'lock'
            with path.open('a') as owner,path.open('a') as waiting:
                fcntl.flock(owner,fcntl.LOCK_EX)
                release=threading.Timer(.03,lambda:fcntl.flock(owner,fcntl.LOCK_UN));release.start()
                try:self.assertTrue(monitor.acquire_update_lock(waiting,wait_seconds=.5))
                finally:release.join()
            self.assertEqual(signal.getitimer(signal.ITIMER_REAL),(0.,0.))

    def test_reload_lock_deadline_leaves_running_episode_owned(self):
        with tempfile.TemporaryDirectory() as directory:
            path=Path(directory)/'lock';handler=signal.getsignal(signal.SIGALRM)
            with path.open('a') as owner,path.open('a') as waiting:
                fcntl.flock(owner,fcntl.LOCK_EX)
                self.assertFalse(monitor.acquire_update_lock(waiting,wait_seconds=.03))
                with self.assertRaises(BlockingIOError):fcntl.flock(waiting,fcntl.LOCK_EX|fcntl.LOCK_NB)
            self.assertEqual(signal.getsignal(signal.SIGALRM),handler)
            self.assertEqual(signal.getitimer(signal.ITIMER_REAL),(0.,0.))

    def test_pause_arriving_while_waiting_prevents_reload(self):
        with tempfile.TemporaryDirectory() as directory:
            root=Path(directory);request=root/'controller-reload-request.json';request.write_text('{"pending":true,"frontend_revisions":{"24":"_review4"}}')
            def acquire(lock):
                (root/'user-pause.json').write_text('{"status":"paused_by_user"}');return True
            with patch.object(monitor,'OUT',root),patch.object(monitor,'BATCH',root),patch.object(monitor,'LOCK',root/'lock'),patch.object(monitor,'acquire_update_lock',side_effect=acquire),patch.object(monitor.subprocess,'run') as run:
                self.assertFalse(monitor.apply_pending_update(False,False));run.assert_not_called()
            self.assertTrue(json.loads(request.read_text())['pending'])
            self.assertFalse((root/'frontend-revisions.json').exists())

    def test_recovery_never_duplicates_owned_work(self):
        for state in ['inactive', 'failed', 'active', 'activating', 'deactivating']:
            self.assertFalse(recovery_decision(state, False, False, True))

    def test_pause_and_completion_prevent_controller_restart(self):
        self.assertFalse(recovery_decision('failed', True, False, False))
        self.assertFalse(recovery_decision('inactive', False, True, False))

    def test_only_stopped_or_failed_controller_is_recovered(self):
        self.assertTrue(recovery_decision('failed', False, False, False))
        self.assertTrue(recovery_decision('inactive', False, False, False))
        for state in ['active', 'activating', 'deactivating', 'unknown']:
            self.assertFalse(recovery_decision(state, False, False, False))

    def test_partial_metric_append_keeps_last_complete_progress(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / 'metrics.jsonl'
            path.write_text('{"environment_steps": 100}\n{"environment_steps":')
            self.assertEqual(latest_metric(path)['environment_steps'], 100)

    def test_missing_timestamp_is_unknown(self):
        self.assertIsNone(age(None, 123))
        self.assertIsNone(age('broken', 123))
        self.assertEqual(age('1970-01-01T00:01:00Z', 123), 63)

    def test_monitor_does_not_trust_a_completion_flag_without_evidence(self):
        done,invalid,checks=monitor.strict_progress({'episodes_done':[19],'episodes':{}},{19,20})
        self.assertEqual(done,[]);self.assertEqual(invalid,[19])
        self.assertTrue(checks[19]['blocking_reasons'])

    def test_code_reload_waits_for_owned_episode(self):
        with tempfile.TemporaryDirectory() as directory:
            root=Path(directory);request=root/'controller-reload-request.json'
            request.write_text(json.dumps({'pending':True}))
            with patch.object(monitor,'OUT',root),patch.object(monitor,'BATCH',root),patch.object(monitor,'LOCK',root/'lock'),patch.object(monitor,'owner_alive',return_value=True),patch.object(monitor.subprocess,'run') as run:
                self.assertFalse(monitor.apply_pending_update(False,False));run.assert_not_called()
            self.assertTrue(json.loads(request.read_text())['pending'])

    def test_code_reload_is_applied_only_without_owned_work(self):
        with tempfile.TemporaryDirectory() as directory:
            root=Path(directory);request=root/'controller-reload-request.json'
            request.write_text(json.dumps({'pending':True}))
            with patch.object(monitor,'OUT',root),patch.object(monitor,'BATCH',root),patch.object(monitor,'LOCK',root/'lock'),patch.object(monitor,'owner_alive',return_value=False),patch.object(monitor,'native_learners',return_value=[]),patch.object(monitor.subprocess,'run') as run:
                self.assertFalse(monitor.apply_pending_update(True,False));run.assert_not_called()
                self.assertTrue(monitor.apply_pending_update(False,False));run.assert_called_once()
            self.assertFalse(json.loads(request.read_text())['pending'])


if __name__ == '__main__':
    unittest.main()
