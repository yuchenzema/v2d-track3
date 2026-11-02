from dataclasses import dataclass
import unittest
from unittest.mock import Mock
import numpy as np
from legacy_source_frame_playback import preserve_source_frames,prepare_script


@dataclass
class Metadata:
    is_resampled:bool=False
    source_frame_playback:bool=False


@dataclass
class Reference:
    _metadata:Metadata
    _fps:float
    positions:np.ndarray
    @property
    def metadata(self):return self._metadata
    @property
    def fps(self):return self._fps


class SourceFramePlaybackTests(unittest.TestCase):
    def test_generated_recording_and_future_import_scripts_share_the_adapter(self):
        from pathlib import Path
        source=(Path(__file__).parent/'runs/episode_000005_newton_20261015/scripts/record_reconstruction.py').read_text()
        for original in [source,'"""Viewer"""\nfrom __future__ import annotations\nvalue=1\n']:
            prepared=prepare_script(original,dict(validated_video_pipeline=True))
            compile(prepared,'generated','exec')
            self.assertIn('install_source_frame_playback()',prepared)
            self.assertEqual(prepare_script(prepared,dict(validated_video_pipeline=True)),prepared)
            self.assertEqual(prepare_script(original,{}),original)
    def test_native_body_identity_accepts_sanitized_spelling_but_not_a_different_object(self):
        from grasp_physics_checks import native_body_id
        names={'track3_mini_dust_pan_object':17}
        self.assertEqual(native_body_id(lambda name:names.get(name,-1),'track3_mini_dust_pan/object'),17)
        with self.assertRaises(AssertionError):native_body_id(lambda name:names.get(name,-1),'track3_white_pot/object')
        with self.assertRaises(AssertionError):native_body_id(lambda name:{'a/b':1,'a_b':2}.get(name,-1),'a/b')

    def test_control_rate_never_changes_source_samples_or_pose_arrays(self):
        poses=np.arange(263*3).reshape(263,3)
        original=Reference(Metadata(),20.,poses)
        loader=Mock(return_value=original)
        result=preserve_source_frames(loader,'source.parquet',control_fps=50.)
        loader.assert_called_once_with('source.parquet',control_fps=None,motion_speed=1.)
        self.assertIs(result.positions,poses)
        self.assertEqual(len(result.positions),263)
        self.assertTrue(result.metadata.source_frame_playback)
        self.assertFalse(result.metadata.is_resampled)
        self.assertFalse(original.metadata.source_frame_playback)
        self.assertEqual(result.fps,50.)

    def test_time_warping_is_rejected_before_loading(self):
        loader=Mock()
        with self.assertRaises(ValueError):preserve_source_frames(loader,'source.parquet',50.,.5)
        loader.assert_not_called()


if __name__=='__main__':unittest.main()
