import json
from pathlib import Path
import tempfile
import unittest
import numpy as np
from improvement_common import digest
from export_reference_diagnostics import ensure


class ExportDiagnosticsTests(unittest.TestCase):
    def test_docker_owned_export_can_be_archived_when_hardlinks_are_forbidden(self):
        from unittest.mock import patch
        from batch_export import archive_file
        with tempfile.TemporaryDirectory() as directory:
            source=Path(directory)/'movie.mp4';target=Path(directory)/'archive.mp4';source.write_bytes(b'actual policy movie')
            with patch('batch_export.os.link',side_effect=PermissionError('protected hardlinks')):
                archive_file(source,target)
            self.assertEqual(target.read_bytes(),source.read_bytes())
            self.assertTrue(source.exists())
    def test_missing_frontend_reports_are_described_without_changing_frozen_poses(self):
        with tempfile.TemporaryDirectory() as directory:
            root=Path(directory);rec=root/'reconstruction';rec.mkdir()
            (root/'episode-config.json').write_text(json.dumps(dict(frames=3,objects=['cup'])))
            camera=np.tile(np.eye(4),(3,1,1));objects=camera[:,None].copy();objects[:,0,0,3]=[0.,.01,.03]
            path=rec/'scene_motion_physical.npz';np.savez(path,camera_to_world=camera,object_to_world=objects);before=digest(path)
            ensure(root);self.assertEqual(digest(path),before)
            report=json.loads((rec/'pose-continuity.json').read_text())
            self.assertAlmostEqual(report['objects']['cup']['translation_step_m_p50_p95_max'][-1],.02)
            self.assertEqual(report['trajectory_sha256'],before)
            camera_report=rec/'camera-stability.json';saved=camera_report.read_bytes();ensure(root)
            self.assertEqual(camera_report.read_bytes(),saved)


if __name__=='__main__':unittest.main()
