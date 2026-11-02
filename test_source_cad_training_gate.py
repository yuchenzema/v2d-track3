import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import numpy as np
import source_cad_training_gate as gate
from improvement_common import digest


class CadTrainingGateTests(unittest.TestCase):
    def test_matching_trajectory_still_requires_final_scene_occlusion_audit(self):
        with tempfile.TemporaryDirectory() as directory:
            base=Path(directory);queue=base/'focused_episodes_20261028';queue.mkdir()
            (queue/'source-first-policy.json').write_text(json.dumps(dict(enabled=True,episodes=[19])))
            root=base/'cad';(root/'reconstruction').mkdir(parents=True)
            train=base/'train';(train/'reconstruction').mkdir(parents=True)
            source=base/'source.mp4';source.write_bytes(b'source identity')
            (root/'episode-config.json').write_text(json.dumps(dict(source=str(source))))
            for p in [root/'reconstruction/cad-source-trajectory.npz',train/'reconstruction/scene_motion_physical.npz']:
                np.savez(p,camera_to_world=np.eye(4)[None],object_to_world=np.eye(4)[None,None])
            report=dict(all_source_frames_checked=True,local_overlap_failed_object_frames=0,unobservable_object_frames=0,
                        run=str(root),trajectory_sha256=digest(root/'reconstruction/cad-source-trajectory.npz'),source_video_sha256=digest(source))
            (queue/'source-cad-status.json').write_text(json.dumps(dict(results={'19':report})))
            with patch.object(gate,'BATCH',base):
                with self.assertRaisesRegex(AssertionError,'Final scene occlusion audit missing'):
                    gate.require_source_cad_alignment(19,train)

    def test_mismatched_training_trajectory_is_rejected(self):
        with tempfile.TemporaryDirectory() as d:
            base=Path(d);queue=base/'focused_episodes_20261028';queue.mkdir()
            (queue/'source-first-policy.json').write_text(json.dumps(dict(enabled=True,episodes=[19])))
            root=base/'cad';(root/'reconstruction').mkdir(parents=True)
            train=base/'train';(train/'reconstruction').mkdir(parents=True)
            source=base/'source.mp4';source.write_bytes(b'source identity')
            (root/'episode-config.json').write_text(json.dumps(dict(source=str(source))))
            M=np.eye(4)[None];np.savez(root/'reconstruction/cad-source-trajectory.npz',camera_to_world=M,object_to_world=M)
            changed=M.copy();changed[0,0,3]=.05
            np.savez(train/'reconstruction/scene_motion_physical.npz',camera_to_world=M,object_to_world=changed)
            report=dict(all_source_frames_checked=True,local_overlap_failed_object_frames=0,unobservable_object_frames=0,
                        run=str(root),trajectory_sha256=digest(root/'reconstruction/cad-source-trajectory.npz'),source_video_sha256=digest(source))
            (queue/'source-cad-status.json').write_text(json.dumps(dict(results={'19':report})))
            with patch.object(gate,'BATCH',base):
                with self.assertRaisesRegex(AssertionError,'Training trajectory differs'):gate.require_source_cad_alignment(19,train)

    def test_missing_overlay_audit_blocks_training(self):
        with tempfile.TemporaryDirectory() as d:
            base=Path(d);queue=base/'focused_episodes_20261028';queue.mkdir()
            (queue/'source-first-policy.json').write_text(json.dumps(dict(enabled=True,episodes=[19])))
            with patch.object(gate,'BATCH',base):
                with self.assertRaisesRegex(AssertionError,'CAD overlay audit missing'):gate.require_source_cad_alignment(19,base)


if __name__=='__main__':unittest.main()
