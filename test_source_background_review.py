import json,tempfile,unittest
from pathlib import Path
import cv2,numpy as np
from improvement_common import digest
from source_background_review import apply_review


class SourceBackgroundReviewTests(unittest.TestCase):
    def fixture(self,root):
        rec=root/'reconstruction';(rec/'frames').mkdir(parents=True);(rec/'masks/1').mkdir(parents=True)
        (root/'episode-config.json').write_text(json.dumps(dict(episode=25,frames=1,objects=['board'])))
        image=rec/'frames/000000.png';cv2.imwrite(str(image),np.full((12,16,3),180,np.uint8))
        mask=rec/'masks/1/000000.png';cv2.imwrite(str(mask),np.full((12,16),255,np.uint8))
        review=dict(episode=25,annotations=[dict(frame=0,object_id=1,source_frame_sha256=digest(image),background_pixels=[[5,6],[6,6]])])
        path=rec/'source-background-review.json';path.write_text(json.dumps(review));return image,mask,path,review

    def test_annotation_survives_repropagation_and_removes_only_explicit_background(self):
        with tempfile.TemporaryDirectory() as d:
            root=Path(d);image,mask,_,_=self.fixture(root);source_sha=digest(image)
            for attempt in range(2):
                cv2.imwrite(str(mask),np.full((12,16),255,np.uint8));report=apply_review(root)
                actual=cv2.imread(str(mask),0);self.assertEqual(np.count_nonzero(actual==0),2)
                self.assertEqual(report['records'][0]['removed_foreground_pixels'],2);self.assertFalse(report['foreground_added'])
                self.assertEqual(digest(image),source_sha)
            before=digest(mask);self.assertEqual(apply_review(root)['records'][0]['removed_foreground_pixels'],0)
            self.assertEqual(digest(mask),before)

    def test_stale_source_binding_rejects_annotation_without_modifying_mask(self):
        with tempfile.TemporaryDirectory() as d:
            root=Path(d);_,mask,path,review=self.fixture(root);before=digest(mask)
            review['annotations'][0]['source_frame_sha256']='wrong';path.write_text(json.dumps(review))
            with self.assertRaisesRegex(RuntimeError,'hash mismatch'):apply_review(root)
            self.assertEqual(digest(mask),before)

    def test_invalid_later_annotation_fails_before_any_partial_mask_changes(self):
        with tempfile.TemporaryDirectory() as d:
            root=Path(d);_,mask,path,review=self.fixture(root);before=digest(mask)
            review['annotations'].append({**review['annotations'][0],'background_pixels':[[-1,0]]});path.write_text(json.dumps(review))
            with self.assertRaisesRegex(RuntimeError,'out of bounds'):apply_review(root)
            self.assertEqual(digest(mask),before)
