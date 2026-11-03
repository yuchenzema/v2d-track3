import json,tempfile,unittest
from pathlib import Path
from unittest.mock import patch
import numpy as np,pyarrow as pa,pyarrow.parquet as pq,trimesh
from source_mask_rasterizer import MeshSilhouette
from source_policy_validation import score


class FullSourceInspectionTests(unittest.TestCase):
    def test_a_one_frame_drop_outside_the_old_sample_is_inspected(self):
        with tempfile.TemporaryDirectory() as directory:
            root=Path(directory);rec=root/'reconstruction';(rec/'mesh_priors').mkdir(parents=True)
            frames=81;intr=dict(fx=500.,fy=500.,cx=512.,cy=384.)
            (root/'episode-config.json').write_text(json.dumps(dict(frames=frames,objects=['object'])))
            (rec/'camera_intrinsics.json').write_text(json.dumps(intr));np.savez(rec/'scene_motion_physical.npz',camera_to_world=np.repeat(np.eye(4)[None],frames,axis=0))
            mesh=trimesh.creation.box(extents=[.1,.1,.04]);mesh.export(rec/'mesh_priors/object.ply')
            pose=np.eye(4);pose[2,3]=.5;renderer=MeshSilhouette(mesh.vertices,mesh.faces,intr);mask=renderer.render(pose);depth=renderer.depth(pose)
            missed=sorted(set(range(frames))-set(np.linspace(0,frames-1,40,dtype=int)))[-1]
            values=dict(object_id=np.zeros(frames,dtype=int),step=np.arange(frames))
            for axis in 'xyz':values['achieved_position_'+axis]=np.full(frames,.5 if axis=='z' else 0.)
            values['achieved_position_x'][missed]=2.
            for axis in 'xyzw':values['achieved_quaternion_'+axis]=np.full(frames,1. if axis=='w' else 0.)
            path=root/'actual.parquet';pq.write_table(pa.table(values),path)
            seen=[]
            def observe(rec,slot,frame):seen.append(frame);return mask,depth,depth>0
            with patch('source_policy_validation.observation',side_effect=observe):result=score(root,path,root)
            self.assertEqual(seen,list(range(frames)));self.assertIn(missed,seen)
            self.assertEqual(result['inspected_object_frames'],frames)
            self.assertGreater(result['source_video_loss'],.01)
            self.assertFalse(result['strict_episode_completion']);self.assertTrue(result['not_kaggle_score'])
