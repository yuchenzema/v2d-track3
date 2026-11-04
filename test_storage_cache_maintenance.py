import json,tempfile,unittest
from pathlib import Path
from datetime import datetime,timedelta,timezone
from unittest.mock import patch
import storage_cache_maintenance as cache


class CompilerCacheTests(unittest.TestCase):
    def test_only_generated_cache_files_are_removed(self):
        with tempfile.TemporaryDirectory() as directory:
            root=Path(directory)
            for name in ['root/.cache/warp/kernel','root/.nv/ComputeCache/kernel','episode/policy.safetensors','root/site-packages/runtime.py']:
                p=root/name;p.parent.mkdir(parents=True,exist_ok=True);p.write_bytes(b'retain scientific artifacts')
            self.assertEqual(len(cache.clear_compiler_caches(root)),2)
            self.assertEqual((root/'episode/policy.safetensors').read_bytes(),b'retain scientific artifacts')
            self.assertTrue((root/'root/site-packages/runtime.py').exists())

    def test_redirected_cache_cannot_delete_artifacts(self):
        with tempfile.TemporaryDirectory() as directory:
            root=Path(directory);data=root/'episode';data.mkdir();(data/'policy').write_bytes(b'keep')
            p=root/'root/.cache';p.mkdir(parents=True);(p/'warp').symlink_to(data,target_is_directory=True)
            with self.assertRaisesRegex(AssertionError,'redirected'):cache.clear_compiler_caches(root)
            self.assertTrue((data/'policy').exists())

    def test_active_and_own_geometry_runtimes_are_never_cleaned(self):
        with tempfile.TemporaryDirectory() as directory:
            base=Path(directory);root=base/'runs/retired';root.mkdir(parents=True);cfg=root/'episode-config.json';cfg.write_text('{}')
            d=dict(State=dict(Status='exited',FinishedAt=(datetime.now(timezone.utc)-timedelta(days=2)).isoformat()),
                Config=dict(Entrypoint=['/bin/sleep'],Cmd=['infinity']),Mounts=[dict(Source=str(root),Destination='/episode')])
            with patch.object(cache,'BASE',base):
                self.assertTrue(cache.retired(d))
                d['State']['FinishedAt']=(datetime.now(timezone.utc)-timedelta(days=2)).strftime('%Y-%m-%dT%H:%M:%S')+'.123456789Z'
                self.assertTrue(cache.retired(d))
                d['State']['Status']='running';self.assertFalse(cache.retired(d));d['State']['Status']='exited'
                cfg.write_text('{"own_geometry_required":true}');self.assertFalse(cache.retired(d));cfg.write_text('{}')
                d['Mounts'].append(dict(Source=str(root),Destination='/root/.cache'));self.assertFalse(cache.retired(d))

    def test_user_pause_prevents_any_container_action(self):
        with tempfile.TemporaryDirectory() as directory:
            base=Path(directory);out=base/'queue';out.mkdir();(base/'user-pause.json').write_text('{"status":"paused_by_user"}')
            with patch.object(cache,'free_bytes',return_value=1),patch.object(cache,'docker') as docker:
                self.assertEqual(cache.reclaim(out),[]);docker.assert_not_called()

    def test_container_is_stopped_even_if_cache_cleanup_fails(self):
        with tempfile.TemporaryDirectory() as directory:
            out=Path(directory);calls=[]
            def docker(*args):
                calls.append(args)
                if args[0]=='ps':return 'abc123def456 v2d-old-newton'
                if args[0]=='inspect':return '[{}]'
                if args[0]=='exec':raise RuntimeError('cleanup failed')
                return ''
            with patch.object(cache,'free_bytes',return_value=1),patch.object(cache,'docker',side_effect=docker),patch.object(cache,'retired',return_value=True):
                with self.assertRaisesRegex(RuntimeError,'cleanup failed'):cache.reclaim(out)
            self.assertIn(('stop','--time','1','abc123def456'),calls)
