import json,tempfile,unittest
from pathlib import Path
from unittest.mock import patch
import cv2
import numpy as np
import fresh_evaluation_frontend as frontend
from public_validation_pipeline import copy_measurement


class FreshEvaluationInputTests(unittest.TestCase):
    def test_docker_owned_measurements_fall_back_from_hardlink_to_copy(self):
        with tempfile.TemporaryDirectory() as directory:
            source=Path(directory)/'stereo.png';target=Path(directory)/'copy.png';source.write_bytes(b'metric source depth')
            with patch('public_validation_pipeline.os.link',side_effect=PermissionError('protected hardlinks')):
                copy_measurement(source,target)
            self.assertEqual(target.read_bytes(),source.read_bytes())

    def test_only_image_measurements_are_reused_not_old_object_poses_or_grasps(self):
        with tempfile.TemporaryDirectory() as directory:
            base=Path(directory);source=base/'runs/old';rec=source/'reconstruction';(rec/'frames').mkdir(parents=True)
            (source/'episode-config.json').write_text(json.dumps(dict(frames=3,objects=['white_pot'],grasps=[dict(start=12)],source='must_not_be_reused')))
            (rec/'video-preparation.json').write_text('{}')
            (rec/'scene_motion_physical.npz').write_bytes(b'old unverified object trajectory')
            (rec/'object_poses').mkdir();(rec/'object_poses/old.json').write_text('{}')
            video=base/'datasets/video_to_data_challenge/track_3/evaluation/videos/chunk-000/observation.images.ego_cam_a/episode_000019.mp4';video.parent.mkdir(parents=True)
            writer=cv2.VideoWriter(str(video),cv2.VideoWriter_fourcc(*'mp4v'),20.,(64,48));self.assertTrue(writer.isOpened())
            for frame in range(3):
                image=np.full((48,64,3),frame*40,np.uint8);writer.write(image);cv2.imwrite(str(rec/f'frames/{frame:06d}.png'),image)
            writer.release()
            with patch.object(frontend,'BASE',base):root,cfg=frontend.prepare(source,19)
            self.assertEqual(cfg['source'],str(video));self.assertEqual(cfg['grasps'],[])
            self.assertFalse((root/'reconstruction/scene_motion_physical.npz').exists())
            self.assertFalse((root/'reconstruction/object_poses').exists())
            self.assertTrue((root/'reconstruction/frames/000002.png').is_file())
            self.assertFalse(cfg['evaluation_ground_truth_used'])
            # A serial repair reuses the fresh frontend as its measurement
            # source; it must not copy/chmod those same cached directories.
            with patch.object(frontend,'BASE',base), patch('shutil.copytree') as copytree:
                resumed,resumed_cfg=frontend.prepare(root,19)
            copytree.assert_not_called()
            self.assertEqual(resumed,root)
            self.assertEqual(resumed_cfg['grasps'],[])
            self.assertTrue((root/'reconstruction/frames/000002.png').is_file())
