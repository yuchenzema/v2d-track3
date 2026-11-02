import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import cv2
import numpy as np

from improvement_common import digest
from source_cap_masks import refine,verify,writable_mask_directory


class SourceCapMaskTests(unittest.TestCase):
    def fixture(self,directory,visible=True,other_owns_cap=False):
        root=Path(directory)/'run';rec=root/'reconstruction'
        (rec/'frames').mkdir(parents=True);(root/'research').mkdir()
        for i in [1,2]:(rec/f'masks/{i}').mkdir(parents=True)
        cfg=dict(episode=24,dataset_split='public',frames=1,objects=['wooden_piece_1','wooden_piece_2'],validated_video_pipeline=True)
        (root/'episode-config.json').write_text(json.dumps(cfg))
        image=np.full((100,100,3),220,np.uint8)
        cv2.rectangle(image,(20,46),(80,85),(60,135,190),-1)
        centers=[[35,40],[50,40],[65,40]]
        if visible:
            for xy in centers:cv2.circle(image,tuple(xy),4,(150,150,150),-1)
        frame=rec/'frames/000000.png';cv2.imwrite(str(frame),image)
        own=np.zeros((100,100),np.uint8);own[46:86,20:81]=255
        other=np.zeros_like(own)
        if other_owns_cap:cv2.circle(other,(65,40),5,255,-1)
        cv2.imwrite(str(rec/'masks/1/000000.png'),own);cv2.imwrite(str(rec/'masks/2/000000.png'),other)
        (rec/'sam2_prompts.json').write_text('{"prompts": []}')
        reviews=Path(directory)/'reviews';reviews.mkdir()
        review=dict(frame=0,object='wooden_piece_1',source_frame_sha256=digest(frame),
                    points=[dict(xy=xy,radius_px=4.) for xy in centers])
        (reviews/'public_000024.json').write_text(json.dumps(dict(episode=24,dataset_split='public',reviews=[review])))
        return root,reviews,own,other

    def test_measured_source_caps_recover_foreground_without_changing_original_source_or_body(self):
        with tempfile.TemporaryDirectory() as directory:
            root,reviews,before,_=self.fixture(directory);rec=root/'reconstruction';source_sha=digest(rec/'frames/000000.png')
            report=refine(root,reviews);after=cv2.imread(str(rec/'masks/1/000000.png'),0)
            self.assertTrue(all(after[40,x] for x in [35,50,65]))
            self.assertTrue(np.all(after[before>0]>0));self.assertEqual(digest(rec/'frames/000000.png'),source_sha)
            self.assertTrue(list((rec/'masks-before-source-caps').rglob('000000.png')))
            self.assertFalse(report['cad_pose_used']);self.assertFalse(report['ground_truth_used'])
            self.assertIsNotNone(verify(root,digest(rec/'sam2_prompts.json'),reviews))

    def test_review_coordinates_without_visible_caps_never_create_foreground(self):
        with tempfile.TemporaryDirectory() as directory:
            root,reviews,before,_=self.fixture(directory,visible=False)
            refine(root,reviews)
            np.testing.assert_array_equal(cv2.imread(str(root/'reconstruction/masks/1/000000.png'),0),before)

    def test_existing_other_instance_labels_cannot_be_stolen_by_a_cap_review(self):
        with tempfile.TemporaryDirectory() as directory:
            root,reviews,_,other=self.fixture(directory,other_owns_cap=True)
            report=refine(root,reviews);rec=root/'reconstruction'
            self.assertEqual(cv2.imread(str(rec/'masks/1/000000.png'),0)[40,65],0)
            np.testing.assert_array_equal(cv2.imread(str(rec/'masks/2/000000.png'),0),other)
            self.assertTrue(any(r.get('existing_other_instance_pixels_skipped',0)>0 for r in report['measurements']))

    def test_changed_masks_or_prompts_invalidate_the_source_cap_binding(self):
        with tempfile.TemporaryDirectory() as directory:
            root,reviews,_,_=self.fixture(directory);rec=root/'reconstruction';refine(root,reviews)
            with self.assertRaisesRegex(RuntimeError,'different source prompts'):verify(root,'changed',reviews)
            mask=cv2.imread(str(rec/'masks/1/000000.png'),0);mask[0,0]=255;cv2.imwrite(str(rec/'masks/1/000000.png'),mask)
            with self.assertRaisesRegex(RuntimeError,'mask evidence changed'):verify(root,digest(rec/'sam2_prompts.json'),reviews)

    def test_container_owned_directory_repair_is_scoped_and_uses_no_gpu_or_recursive_chown(self):
        with tempfile.TemporaryDirectory() as directory:
            root,reviews,_,_=self.fixture(directory)
            with patch('source_cap_masks.os.access',return_value=False),patch('batch_native._docker') as docker:
                writable_mask_directory(root,root/'reconstruction/masks/1')
            args=docker.call_args.args[0]
            self.assertEqual(args[-1],'/episode/reconstruction/masks/1')
            self.assertNotIn('-R',args);self.assertNotIn('--gpus',args)
            self.assertEqual(args[args.index('--entrypoint')+1],'/bin/chown')
