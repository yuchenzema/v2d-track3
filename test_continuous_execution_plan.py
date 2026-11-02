"""Verify that all requested clips remain in automatic execution after audit."""
import unittest
import tempfile
from pathlib import Path
from unittest.mock import patch
import continuous_human_match_20261023 as controller
from continuous_human_match_20261023 import execution_plan,EXPECTED

class AutomaticExecutionTests(unittest.TestCase):
    def test_focused_scope_parks_all_other_episodes(self):
        focus=[19,20,25,28,30,32,33,36]
        state={'require_verified_match_before_next':True,'episodes':{}}
        with patch.object(controller,'FOCUS',focus),patch.object(controller,'SERIAL_ORDER',focus):
            plan=execution_plan(state,{})
            self.assertEqual({int(ep) for ep,row in plan.items() if row['automatic']},set(focus))
            self.assertTrue(all(row['status']=='parked_outside_focus' for ep,row in plan.items() if int(ep) not in focus))
            self.assertEqual(controller.required_episode(state),19)
    def test_all_twenty_are_registered_while_priority_policy_trains(self):
        state={'episodes':{str(ep):{'passes_strict_local_checks':False,'recommended_action':'reconstruct'} for ep in EXPECTED}}
        plan=execution_plan(state,{'status':'running','active':{'episode':19,'stage':'train_rollout'}})
        self.assertEqual(set(map(int,plan)),{1,3,5,6,8,10,14,15,16,19,20,25,28,30,32,33,35,36,37,38})
        self.assertEqual(plan['19']['status'],'running_priority_pipeline')
        self.assertTrue(all(plan[str(ep)]['status']=='queued_priority_pipeline' for ep in [20,30,33]))
        self.assertTrue(all(plan[str(ep)]['status']=='queued_corrective_cycle' for ep in EXPECTED if ep not in [19,20,30,33]))
        self.assertTrue(all(row['automatic'] and row['recheck_after_every_rollout'] for row in plan.values()))
    def test_finished_training_does_not_close_unmatched_episode(self):
        state={'episodes':{'19':{'passes_strict_local_checks':False,'recommended_action':'train_rollout'}}}
        self.assertEqual(execution_plan(state,{'status':'completed'})['19']['status'],'queued_corrective_cycle')
    def test_active_corrective_episode_remains_visible(self):
        state={'active_optimization':{'episode':28}}
        plan=execution_plan(state,{'status':'completed'})
        self.assertEqual(plan['28']['status'],'running_corrective_cycle')
        self.assertFalse(any(row['exact_human_match_verified'] for row in plan.values()))
    def test_sam3d_switch_holds_new_jobs_but_keeps_current_job_visible(self):
        plan=execution_plan({'dispatch_hold':{'method':'SAM 3D Objects'},'active_optimization':{'episode':32}}, {'status':'completed'})
        self.assertEqual(plan['32']['status'],'running_corrective_cycle')
        self.assertTrue(all(row['status']=='waiting_sam3d' for ep,row in plan.items() if ep!='32'))
    def test_independently_queued_episode20_is_visible(self):
        plan=execution_plan({'external_optimization':{'episode':20}}, {'status':'completed'})
        self.assertEqual(plan['20']['status'],'running_priority_pipeline')
        self.assertTrue(all(row['status']=='queued_corrective_cycle' for ep,row in plan.items() if ep!='20'))
    def test_new_generation_never_overwrites_existing_run_or_request(self):
        with tempfile.TemporaryDirectory() as d:
            base=Path(d);out=base/'queue';(out/'jobs').mkdir(parents=True)
            (base/'runs/episode_000020_newton_human_match_20261023_g001').mkdir(parents=True)
            (out/'jobs/episode_000020-g002.json').touch()
            with patch.object(controller,'BASE',base),patch.object(controller,'OUT',out):
                self.assertEqual(controller.next_generation(20,1),3)

if __name__=='__main__':unittest.main()
