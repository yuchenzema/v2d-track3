import unittest
import tempfile,json
from pathlib import Path
import cv2
import numpy as np
from source_mask_quality import contamination,repair_points,merge_repair_prompt,audit,inspect_and_repair,reinforce_foreground


class MaskLeakageTests(unittest.TestCase):
    def test_missing_foreground_is_reconditioned_from_matching_rgb_without_painting_mask(self):
        image=np.full((40,40,3),210,np.uint8);image[20,24]=[0,0,0]
        anchor=dict(x=20.,y=20.);prompts=dict(prompts=[dict(object_id=1,frame_index=0,points=[anchor],point_labels=[1])])
        added=reinforce_foreground(prompts,1,0,image,[anchor])
        self.assertGreater(added,0);row=prompts['prompts'][0]
        self.assertIn(anchor,row['points']);self.assertNotIn(dict(x=24.,y=20.),row['points'])
        self.assertEqual(reinforce_foreground(prompts,1,0,image,[anchor]),0)
        self.assertEqual(row['point_labels'],[1]*len(row['points']))
    def coloured_hole_fixture(self,root):
        rec=root/'reconstruction';(rec/'frames').mkdir(parents=True);(rec/'masks/1').mkdir(parents=True)
        (root/'episode-config.json').write_text(json.dumps(dict(frames=1,objects=['mini_sweeper'])))
        cv2.imwrite(str(rec/'frames/000000.png'),np.full((128,128,3),(40,110,160),np.uint8))
        mask=np.zeros((128,128),np.uint8);mask[20:110,20:110]=255;cv2.imwrite(str(rec/'masks/1/000000.png'),mask)
        np.savez(rec/'hand_measurements.npz',observed=np.zeros((1,2),bool),uv=np.zeros((1,2,21,2)))
        prompt=dict(prompts=[dict(object_id=1,frame_index=0,points=[dict(x=35,y=35),dict(x=65,y=65)],point_labels=[1,0])])
        for name in ['sam2_prompts.json','source-prompt-review.json']:(rec/name).write_text(json.dumps(prompt))
        return rec,mask

    def test_coloured_object_cannot_fill_reviewed_handle_hole(self):
        with tempfile.TemporaryDirectory() as directory:
            root=Path(directory);rec,mask=self.coloured_hole_fixture(root)
            result=audit(root);self.assertEqual(result['failure_count'],1)
            self.assertEqual(result['failures'][0]['included_reviewed_background_points'],[dict(x=65,y=65)])
            self.assertFalse(result['failures'][0]['contaminated'])
            mask[60:70,60:70]=0;cv2.imwrite(str(rec/'masks/1/000000.png'),mask)
            self.assertEqual(audit(root)['failure_count'],0)

    def test_hole_repair_reruns_complete_prompt_without_inventing_white_part(self):
        with tempfile.TemporaryDirectory() as directory:
            root=Path(directory);rec,mask=self.coloured_hole_fixture(root);original=json.loads((rec/'sam2_prompts.json').read_text());calls=[]
            def rerun():
                calls.append(1);mask[60:70,60:70]=0;cv2.imwrite(str(rec/'masks/1/000000.png'),mask)
            inspect_and_repair(root,rerun)
            self.assertEqual(len(calls),1)
            self.assertEqual(json.loads((rec/'sam2_prompts.json').read_text())['prompts'],original['prompts'])

    def test_unrepaired_coloured_hole_still_blocks_after_retries(self):
        with tempfile.TemporaryDirectory() as directory:
            root=Path(directory);self.coloured_hole_fixture(root);calls=[]
            with self.assertRaisesRegex(RuntimeError,'foreground/background-hint failure'):
                inspect_and_repair(root,lambda:calls.append(1))
            self.assertEqual(len(calls),2)

    def test_missing_visible_handle_hint_fails_even_when_colour_check_passes(self):
        with tempfile.TemporaryDirectory() as directory:
            root=Path(directory);rec=root/'reconstruction';(rec/'frames').mkdir(parents=True);(rec/'masks/1').mkdir(parents=True)
            (root/'episode-config.json').write_text(json.dumps(dict(frames=1,objects=['white_pot'])))
            image=np.full((160,160,3),150,np.uint8);image[40:90,40:90]=220;image[120:140,120:140]=220
            cv2.imwrite(str(rec/'frames/000000.png'),image);mask=np.zeros((160,160),np.uint8);mask[40:90,40:90]=255;cv2.imwrite(str(rec/'masks/1/000000.png'),mask)
            np.savez(rec/'hand_measurements.npz',observed=np.zeros((1,2),bool),uv=np.zeros((1,2,21,2)))
            (rec/'source-prompt-review.json').write_text(json.dumps(dict(prompts=[dict(object_id=1,frame_index=0,points=[dict(x=65,y=65),dict(x=130,y=130)],point_labels=[1,1])])))
            self.assertFalse(contamination(image,mask)['contaminated'])
            result=audit(root);self.assertEqual(result['failure_count'],1)
            self.assertEqual(result['failures'][0]['missing_reviewed_foreground_points'],[dict(x=130,y=130)])

    def test_repair_keeps_handle_and_other_frame_conditioning_points(self):
        prompts=dict(prompts=[dict(object_id=1,frame_index=0,points=[dict(x=50,y=50),dict(x=80,y=120)],point_labels=[1,1]),dict(object_id=1,frame_index=50,points=[dict(x=30,y=40)],point_labels=[1])])
        merge_repair_prompt(prompts,1,0,dict(points=[dict(x=45,y=45),dict(x=81,y=120),dict(x=140,y=140)],point_labels=[1,0,0]))
        row=prompts['prompts'][0]
        self.assertIn(dict(x=80,y=120),row['points']);self.assertEqual(row['point_labels'][:2],[1,1])
        self.assertNotIn(dict(x=81,y=120),row['points']);self.assertIn(dict(x=140,y=140),row['points'])
        self.assertEqual(len(prompts['prompts']),2);self.assertEqual(prompts['prompts'][1]['frame_index'],50)

    def image(self):
        hsv=np.zeros((160,160,3),np.uint8);hsv[:]=[15,190,160];hsv[40:90,40:90]=[0,0,220]
        return cv2.cvtColor(hsv,cv2.COLOR_HSV2BGR)

    def test_white_pot_mask_including_table_is_detected(self):
        row=contamination(self.image(),np.ones((160,160),np.uint8)*255)
        self.assertTrue(row['contaminated'])

    def test_correct_white_foreground_is_not_flagged(self):
        mask=np.zeros((160,160),np.uint8);mask[40:90,40:90]=255
        self.assertFalse(contamination(self.image(),mask)['contaminated'])

    def test_repair_prompt_uses_white_foreground_and_actual_table_negative(self):
        prompt=repair_points(self.image(),np.ones((160,160),np.uint8)*255,[])
        positive=prompt['points'][0];self.assertTrue(40<=positive['x']<90 and 40<=positive['y']<90)
        self.assertEqual(prompt['point_labels'],[1,0])
