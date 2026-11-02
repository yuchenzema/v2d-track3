import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import evaluation20_terminal as terminal
import source_cad_queue as queue
from improvement_common import digest


class TerminalAutomationTests(unittest.TestCase):
    def test_finished_training_does_not_hide_failed_bound_grasp(self):
        with tempfile.TemporaryDirectory() as directory:
            out=Path(directory)/'queue';out.mkdir();run=Path(directory)/'pilot'
            (run/'flash_chord_coherent_actor_transfer').mkdir(parents=True)
            (run/'flash_chord_coherent_actor_transfer/TRAINING_COMPLETED').write_text('done')
            terminal.write(out/'experimental-training-status.json',dict(run=str(run),status='pilot_finished'))
            terminal.write(run/'episode-config.json',dict(episode=38))
            strict=dict(checkpoint_sha256='actor',contact_validation_failed=True,failed_frame_count=263,
                official_render_binding=dict(official_trajectory_sha256='rollout'),
                post_training_grasp_acceptance=dict(policy_inference=True,checkpoint_sha256='actor',
                    official_trajectory_sha256='rollout',all_grasps_passed=False))
            terminal.write(run/'research/strict-human-match.json',strict)
            result=terminal.pilot_validation(out)
            self.assertTrue(result['training_finished'])
            self.assertIs(result['policy_grasp_passed'],False)
            self.assertFalse(result['strict_completion_granted'])
            strict['post_training_grasp_acceptance']['checkpoint_sha256']='another actor'
            terminal.write(run/'research/strict-human-match.json',strict)
            self.assertIsNone(terminal.pilot_validation(out)['policy_grasp_passed'])

    def test_blocked_export_keeps_last_episode_and_completed_training_steps(self):
        with tempfile.TemporaryDirectory() as directory:
            out=Path(directory)/'queue';out.mkdir();run=Path(directory)/'pilot'
            folder=run/'flash_chord_coherent_actor_transfer';folder.mkdir(parents=True)
            (folder/'metrics.jsonl').write_text('{"environment_steps":1280000}\n')
            (folder/'TRAINING_COMPLETED').write_text('done')
            terminal.write(run/'episode-config.json',dict(episode=38))
            terminal.write(out/'status.json',dict(status='experimental_policy_blocked',current_episode=None,active_optimization=None))
            terminal.write(out/'experimental-training-status.json',dict(status='pilot_blocked',run=str(run),failure_summary='source format error'))
            result=terminal.snapshot(out)
            self.assertEqual(result['active_episode'],38)
            self.assertEqual(result['training_steps'],1280000)
            self.assertIn('訓練後',result['stage'])

    def test_finished_training_remains_visible_while_another_episode_does_cad(self):
        with tempfile.TemporaryDirectory() as directory:
            out=Path(directory)/'queue';out.mkdir();run=Path(directory)/'pilot'
            (run/'flash_chord_coherent_actor_transfer').mkdir(parents=True)
            (run/'flash_chord_coherent_actor_transfer/metrics.jsonl').write_text('{"environment_steps":1280000}\n')
            terminal.write(run/'episode-config.json',dict(episode=38))
            terminal.write(out/'experimental-training-status.json',dict(status='pilot_finished',run=str(run)))
            terminal.write(out/'status.json',dict(status='source_cad_alignment',current_episode=19))
            terminal.write(out/'source-cad-status.json',dict(active_episode=19,run=str(Path(directory)/'cad19')))
            result=terminal.snapshot(out)
            self.assertEqual(result['active_episode'],19)
            self.assertEqual(result['training_steps'],0)
            self.assertEqual(result['recorded_training_steps_by_episode'],{'38':1280000})
    def test_completed_pilot_keeps_actual_steps_after_learner_exits(self):
        with tempfile.TemporaryDirectory() as directory:
            out=Path(directory)/'queue';out.mkdir();run=Path(directory)/'pilot'
            (run/'flash_chord_coherent_actor_transfer').mkdir(parents=True)
            (run/'flash_chord_coherent_actor_transfer/metrics.jsonl').write_text('{"environment_steps":1280000}\n')
            terminal.write(out/'status.json',dict(status='experimental_policy_validation',current_episode=38,
                active_optimization=dict(run=str(run),experimental_only=True)))
            result=terminal.snapshot(out)
            self.assertEqual(result['training_steps'],1280000)
            self.assertEqual(result['learner_pids'],[])
            self.assertIn('1280000',terminal.render(result))

    def test_pilot_failure_remains_visible_when_controller_returns_to_cad(self):
        with tempfile.TemporaryDirectory() as directory:
            out=Path(directory)
            terminal.write(out/'status.json',dict(status='source_cad_alignment',current_episode=5))
            terminal.write(out/'source-cad-status.json',dict(active_episode=5,stage='CAD_geometry_and_source_pose_fit'))
            terminal.write(out/'experimental-training-status.json',dict(status='pilot_blocked',latest_failure='retarget file missing'))
            self.assertIn('retarget file missing',terminal.snapshot(out)['blocking_failure'])

    def test_overlay_and_historical_health_claims_cannot_become_strict_completion(self):
        with tempfile.TemporaryDirectory() as directory:
            out=Path(directory)
            terminal.write(out/'source-cad-status.json',dict(inspected_count=20,
                results={str(ep):dict(all_source_frames_checked=True,
                    local_overlap_failed_object_frames=0,unobservable_object_frames=0)
                    for ep in terminal.EPISODES}))
            terminal.write(out/'health-monitor.json',dict(episodes_done=terminal.EPISODES,
                native_learner_pids=[999999999],environment_steps=30000000))
            result=terminal.snapshot(out)
            self.assertEqual(result['inspected'],20)
            self.assertEqual(result['source_locally_passed'],20)
            self.assertEqual(result['strict_completion_count'],0)
            self.assertEqual(result['training_steps'],0)
            self.assertIn('沒有 policy',terminal.render(result))

    def test_configure_excludes_public_and_disables_model_delivery_without_changing_thresholds(self):
        with tempfile.TemporaryDirectory() as directory:
            out=Path(directory)
            terminal.write(out/'validated-pipeline-policy.json',dict(enabled=True,
                episodes=terminal.EPISODES,minimum_public_auc=.9,wrist_residual_limit_m=.015))
            terminal.write(out/'session-updates.json',dict(enabled=True,thread_id='preserved'))
            terminal.configure(out)
            policy=terminal.read(out/'validated-pipeline-policy.json')
            self.assertFalse(policy['public_validation_required'])
            self.assertEqual(policy['minimum_public_auc'],.9)
            self.assertEqual(policy['wrist_residual_limit_m'],.015)
            self.assertFalse(terminal.read(out/'session-updates.json')['enabled'])
            self.assertEqual(terminal.read(out/'source-first-policy.json')['episodes'],terminal.EPISODES)

    def test_resume_is_explicit_and_existing_pause_is_honored(self):
        with tempfile.TemporaryDirectory() as directory:
            out=Path(directory)/'out';out.mkdir()
            terminal.write(out/'validated-pipeline-policy.json',dict(enabled=True,episodes=terminal.EPISODES))
            terminal.write(out.parent/'user-pause.json',dict(status='paused_by_user'))
            with patch.object(terminal,'systemctl') as systemctl:
                with self.assertRaisesRegex(RuntimeError,'pause'):
                    terminal.start_services(out)
                self.assertFalse(any(terminal.CONTROLLER in call.args for call in systemctl.call_args_list))
                terminal.start_services(out,resume=True)
                self.assertEqual(terminal.read(out.parent/'user-pause.json')['status'],'resumed_by_user')

    def test_failed_frame_csv_includes_failures_only(self):
        with tempfile.TemporaryDirectory() as directory:
            out=Path(directory)
            terminal.write(out/'frames.json',[dict(frame=1,object='cup',local_overlap_pass=False),
                                             dict(frame=2,object='cup',local_overlap_pass=True)])
            terminal.write(out/'source-cad-status.json',dict(results={'1':dict(frames_report=str(out/'frames.json'))}))
            text=terminal.export_failures(out).read_text()
            self.assertIn('1,1,cup',text)
            self.assertNotIn('1,2,cup',text)

    def test_changed_algorithm_allows_retry_but_same_failed_attempt_does_not(self):
        with tempfile.TemporaryDirectory() as directory:
            base=Path(directory);script=base/'algorithm.py';script.write_text('old')
            report=dict(algorithm_files_sha256={'algorithm.py':digest(script)})
            with patch.object(queue,'BASE',base):
                self.assertEqual(queue.changed_repair_inputs(report),[])
                script.write_text('repaired')
                self.assertEqual(queue.changed_repair_inputs(report),['algorithm.py'])
                attempted=queue.algorithm_snapshot(report)
                self.assertEqual(queue.changed_repair_inputs(report,attempted),[])
                script.write_text('second repair')
                self.assertEqual(queue.changed_repair_inputs(report,attempted),['algorithm.py'])

    def test_monitor_reports_immediately_then_after_ten_minutes_without_real_wait(self):
        with tempfile.TemporaryDirectory() as directory:
            out=Path(directory);clock=[0.]
            def tick(seconds):
                if clock[0]>=600:
                    raise KeyboardInterrupt()
                clock[0]+=seconds
            with patch.object(terminal,'OUT',out), \
                 patch.object(terminal,'start_services') as start, \
                 patch.object(terminal,'systemctl'), \
                 patch.object(terminal.time,'monotonic',side_effect=lambda:clock[0]), \
                 patch.object(terminal.time,'sleep',side_effect=tick), \
                 patch('sys.argv',['evaluation20_terminal.py']), patch('builtins.print'):
                terminal.main()
            start.assert_called_once_with(out,resume=False)
            self.assertEqual((out/'terminal-progress.log').read_text().count('本機進度'),2)
            self.assertTrue((out/'terminal-progress.json').is_file())


if __name__=='__main__':unittest.main()
