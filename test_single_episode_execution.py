import multiprocessing as mp
import os
from pathlib import Path
import queue
import tempfile
import unittest
from unittest.mock import patch
import episode_execution_guard as guard
import continuous_human_match_20261023 as controller
from batch_native import training_capacity

def worker(episode,entered,release,started):
    started.set()
    with guard.episode_slot(Path('/tmp')/str(episode),episode):
        entered.put(episode)
        release.wait(5)

class EpisodeExclusionTests(unittest.TestCase):
    def test_another_episode_waits_until_entire_pipeline_releases_slot(self):
        ctx=mp.get_context('fork');events=ctx.Queue();release1=ctx.Event();release2=ctx.Event();started1=ctx.Event();started2=ctx.Event()
        with tempfile.TemporaryDirectory() as d,patch.object(guard,'LOCK',Path(d)/'lock'),patch.object(guard,'OWNER',Path(d)/'owner.json'),patch.object(guard,'available_ram_bytes',return_value=16*1024**3):
            first=ctx.Process(target=worker,args=(20,events,release1,started1));second=ctx.Process(target=worker,args=(30,events,release2,started2))
            try:
                first.start();self.assertEqual(events.get(timeout=2),20)
                second.start();self.assertTrue(started2.wait(2))
                with self.assertRaises(queue.Empty):events.get(timeout=.2)
                release1.set();self.assertEqual(events.get(timeout=2),30);release2.set()
                first.join(2);second.join(2);self.assertEqual((first.exitcode,second.exitcode),(0,0))
            finally:
                for process in [first,second]:
                    if process.is_alive():process.terminate();process.join(2)
    def test_pid_reuse_does_not_keep_old_reservation(self):
        self.assertTrue(guard.owner_alive({'pid':os.getpid(),'process_signature':guard.process_signature(os.getpid())}))
        self.assertFalse(guard.owner_alive({'pid':os.getpid(),'process_signature':'old-command'}))
        self.assertTrue(guard.owner_alive({'pid':None,'protected_children':[{'pid':os.getpid(),'process_signature':guard.process_signature(os.getpid())}]}))
    def test_training_limit_has_gpu_and_host_memory_headroom(self):
        maximum,minimum,ram_ok=training_capacity(Path('/tmp/episode'),7500,1)
        self.assertEqual(maximum,1);self.assertEqual(minimum,11500)
    def test_active_episode_prevents_audits_and_launches_for_other_episodes(self):
        class StopAfterWaiting(Exception):pass
        with tempfile.TemporaryDirectory() as d,patch.object(controller,'OUT',Path(d)),patch.object(controller,'read',return_value={}),patch.object(controller,'external_optimization',return_value={'episode':20}),patch.object(controller,'import_external_result'),patch.object(controller,'emit'),patch.object(controller,'snapshot') as snapshot,patch.object(controller.subprocess,'Popen') as launch,patch.object(controller.time,'sleep',side_effect=StopAfterWaiting):
            with self.assertRaises(StopAfterWaiting):controller.main()
            snapshot.assert_not_called();launch.assert_not_called()

if __name__=='__main__':unittest.main()
