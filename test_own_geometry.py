import json
from pathlib import Path
import tempfile
import unittest
import numpy as np
from own_geometry import reconstruct_surface, reject_forbidden_path, validate


class OwnGeometryTests(unittest.TestCase):
    def test_measured_surface_and_projection_survive_closed_completion(self):
        rgb=np.full((80,100,3),180,np.uint8);mask=np.zeros((80,100),bool);mask[20:60,25:75]=True
        depth=np.full((80,100),.5,np.float32);depth[35:37,40:45]=0
        intr=dict(fx=120.,fy=120.,cx=50.,cy=40.)
        mesh,pose,proof=reconstruct_surface(rgb,mask,depth,intr,face_budget=600)
        self.assertTrue(mesh.is_watertight);self.assertLessEqual(len(mesh.faces),600)
        points=mesh.vertices+pose[:3,3]
        uv=points[:,:2]/points[:,2:]*120+[50,40]
        self.assertGreaterEqual(uv[:,0].min(),24);self.assertLessEqual(uv[:,0].max(),75)
        self.assertGreater(points[:,2].min(),.49)
        self.assertGreater(proof['estimated_depth_fraction'],0)
        self.assertFalse(proof['hidden_shape_recovered'])

    def test_missing_metric_depth_does_not_invent_scale(self):
        with self.assertRaisesRegex(ValueError,'Insufficient metric'):
            reconstruct_surface(np.zeros((40,40,3),np.uint8),np.ones((40,40),bool),np.zeros((40,40)),dict(fx=80,fy=80,cx=20,cy=20))

    def test_public_geometry_and_symlink_paths_are_rejected(self):
        with tempfile.TemporaryDirectory() as directory:
            base=Path(directory);forbidden=base/'public/mesh/object.ply';forbidden.parent.mkdir(parents=True);forbidden.write_bytes(b'scan')
            alias=base/'renamed.ply';alias.symlink_to(forbidden)
            for path in [forbidden,alias]:
                with self.assertRaisesRegex(ValueError,'forbidden'):reject_forbidden_path(path)

    def test_historical_scan_run_is_ineligible_without_own_proof(self):
        with tempfile.TemporaryDirectory() as directory:
            with self.assertRaisesRegex(ValueError,'Missing own-reconstruction'):validate(Path(directory))
