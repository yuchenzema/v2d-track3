import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import cv2
import numpy as np

import fresh_evaluation_frontend as frontend


class MonocularDepthCacheRestoreTests(unittest.TestCase):
    def setUp(self):
        self.temp=tempfile.TemporaryDirectory();self.addCleanup(self.temp.cleanup)
        self.base=Path(self.temp.name);self.root=self.base/'new';self.rec=self.root/'reconstruction'
        self.rec.mkdir(parents=True);(self.root/'logs').mkdir()
        (self.rec/'camera_intrinsics.json').write_text(json.dumps(dict(width=16,height=12)))
        prior=self.base/'runs/episode_000005_newton_20261015';command=prior/'research/moge-command.json'
        command.parent.mkdir(parents=True)
        command.write_text(json.dumps(['docker','run','-v',str(prior)+':/episode','moge',
            '--depth_folder','/episode/reconstruction/depth','--batch_size','4']))
        self.cfg=dict(frames=3)

    def populate(self,folder,frames=range(3)):
        folder.mkdir(parents=True,exist_ok=True)
        for frame in frames:cv2.imwrite(str(folder/f'{frame:06d}.png'),np.full((12,16),100+frame,np.uint16))

    def run_check(self,regenerate=True):
        def generate(command,log):
            self.assertIn(str(self.root)+':/episode',command)
            self.assertEqual(command[command.index('--depth_folder')+1],'/episode/reconstruction/depth_moge')
            self.assertEqual(command[command.index('--batch_size')+1],'1')
            if regenerate:self.populate(self.rec/'depth_moge')
        with patch.object(frontend,'BASE',self.base),patch('batch_worker.docker',side_effect=generate) as docker:
            frontend.ensure_monocular_depth(self.root,self.cfg)
            return docker.call_count

    def test_retained_last_depth_frame_does_not_hide_missing_middle_frames(self):
        self.populate(self.rec/'depth_moge',[0,2]);self.assertEqual(self.run_check(),1)
        report=json.loads((self.rec/'monocular-depth-cache.json').read_text())
        self.assertTrue(report['all_source_depth_frames_verified']);self.assertTrue(report['regenerated'])

    def test_regeneration_detaches_sparse_historical_symlink_without_writing_old_measurements(self):
        original=self.base/'historical';self.populate(original,[0,2]);before={p.name:p.read_bytes() for p in original.iterdir()}
        (self.rec/'depth_moge').symlink_to(original)
        self.assertEqual(self.run_check(),1);self.assertFalse((self.rec/'depth_moge').is_symlink())
        self.assertEqual({p.name:p.read_bytes() for p in original.iterdir()},before)
        self.assertEqual(len(list((self.rec/'depth_moge').glob('*.png'))),3)

    def test_full_historical_raw_depth_is_reused_read_only_and_fusion_output_is_local(self):
        original=self.base/'historical';self.populate(original);(self.rec/'depth').symlink_to(original)
        self.assertEqual(self.run_check(),0)
        self.assertTrue((self.rec/'depth_moge').is_symlink());self.assertFalse((self.rec/'depth').is_symlink())
        self.assertTrue((self.rec/'depth').is_dir());self.assertEqual(len(list(original.glob('*.png'))),3)

    def test_existing_fusion_output_symlink_cannot_write_back_to_historical_cache(self):
        original=self.base/'historical';self.populate(original);self.populate(self.rec/'depth_moge')
        (self.rec/'depth').symlink_to(original)
        self.assertEqual(self.run_check(),0);self.assertFalse((self.rec/'depth').is_symlink())
        self.assertEqual(list((self.rec/'depth').iterdir()),[])

    def test_incomplete_regeneration_fails_without_a_verification_marker(self):
        self.populate(self.rec/'depth_moge',[2])
        with self.assertRaisesRegex(RuntimeError,'every source-frame depth'):self.run_check(regenerate=False)
        self.assertFalse((self.rec/'monocular-depth-cache.json').exists())

    def test_invalid_depth_shape_is_regenerated_even_if_all_filenames_exist(self):
        self.populate(self.rec/'depth_moge')
        cv2.imwrite(str(self.rec/'depth_moge/000001.png'),np.zeros((2,2),np.uint16))
        self.assertEqual(self.run_check(),1)


if __name__=='__main__':unittest.main()
