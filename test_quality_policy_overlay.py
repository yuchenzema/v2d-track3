import json,tempfile,unittest
from pathlib import Path
from unittest.mock import patch
import cv2,numpy as np,pyarrow as pa,pyarrow.parquet as pq,trimesh
from quality_policy_overlay import render
from improvement_common import digest,save


class OfficialOverlayTests(unittest.TestCase):
    def test_video_uses_achieved_states_and_preserves_every_source_frame(self):
        with tempfile.TemporaryDirectory() as directory:
            root=Path(directory);source=root/'source';rec=source/'reconstruction';(rec/'frames').mkdir(parents=True);(root/'reconstruction/mesh_priors').mkdir(parents=True)
            count=4;poses=np.repeat(np.eye(4)[None,None],count,axis=0);poses[:,:,2,3]=.5
            save(root/'episode-config.json',dict(frames=count,fps=20.,objects=['object'],fast_source_run=str(source)))
            save(rec/'camera_intrinsics.json',dict(fx=500.,fy=500.,cx=512.,cy=384.))
            np.savez(rec/'scene_motion_physical.npz',camera_to_world=np.repeat(np.eye(4)[None],count,axis=0))
            np.savez(root/'reconstruction/scene_motion_physical.npz',object_to_world=poses)
            trimesh.creation.box(extents=[.1,.1,.04]).export(root/'reconstruction/mesh_priors/object.ply')
            for frame in range(count):cv2.imwrite(str(rec/f'frames/{frame:06d}.png'),np.full((384,512,3),60,np.uint8))
            data=dict(object_body_name=['object']*count,step=np.arange(count),reference_frame=np.arange(count))
            for axis in 'xyz':data['achieved_position_'+axis]=np.full(count,.5 if axis=='z' else .1 if axis=='x' else 0.)
            for axis in 'xyzw':data['achieved_quaternion_'+axis]=np.full(count,1. if axis=='w' else 0.)
            trajectory=root/'actual.parquet';pq.write_table(pa.table(data),trajectory)
            save(root/'research/fast-submission-result.json',dict(frozen_input_manifest_sha256='frozen',actual_training_steps=6400))
            item=dict(checkpoint_sha256='abc',trajectory=str(trajectory),trajectory_sha256=digest(trajectory))
            with patch('own_submission_queue.eligible',return_value=item):record=render(root);cached=render(root)
            self.assertEqual(record,cached);self.assertEqual(record['identity']['source_frame_count'],count)
            self.assertTrue(record['source_clock_preserved']);self.assertFalse(record['strict_episode_completion'])
            capture=cv2.VideoCapture(record['video']);decoded=[]
            while True:
                ok,image=capture.read()
                if not ok:break
                decoded.append(image)
            capture.release();self.assertEqual(len(decoded),count)
            self.assertGreater(np.abs(decoded[0][32:,:512].astype(float)-decoded[0][32:,512:].astype(float)).sum(),1000)
