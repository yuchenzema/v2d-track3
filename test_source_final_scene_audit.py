from pathlib import Path
import tempfile
import unittest
import cv2
import numpy as np
from source_final_scene_audit import audit
from source_cad_alignment import object_visibility,metrics,qualifies


class FinalSceneAuditTests(unittest.TestCase):
    def check_scene(self,front_depth,old_pass):
        with tempfile.TemporaryDirectory() as directory:
            root=Path(directory);rec=root/'reconstruction';(rec/'frames').mkdir(parents=True)
            cv2.imwrite(str(rec/'frames/000000.png'),np.zeros((768,1024,3),np.uint8))
            top=np.zeros((384,512),bool);top[:192]=True
            for body,mask in [(1,~top),(2,top)]:
                (rec/f'masks/{body}').mkdir(parents=True)
                cv2.imwrite(str(rec/f'masks/{body}/000000.png'),cv2.resize(mask.astype(np.uint8)*255,(1024,768),interpolation=cv2.INTER_NEAREST))
            class Renderer:
                def __init__(self,mask,depth):self.mask=mask;self.z=depth
                def render(self,pose):return self.mask.copy()
                def depth(self,pose):return self.mask.astype(np.float32)*self.z
            renderers={'back':Renderer(np.ones_like(top),.6),'front':Renderer(top,front_depth)}
            motion={'camera_to_world':np.eye(4)[None], 'object_to_world':np.repeat(np.eye(4)[None,None],2,axis=1)}
            before={name:values.copy() for name,values in motion.items()}
            records=[dict(frame=0,object=name,iou=.5,local_overlap_pass=old_pass) for name in renderers]
            result=audit(root,dict(frames=1,objects=list(renderers)),motion,renderers,records,{}, {},{},
                         lambda hands,frame:np.zeros((768,1024),np.uint8),object_visibility,metrics,qualifies)
            for name in before:np.testing.assert_array_equal(motion[name],before[name])
            return result,{row['object']:row for row in records}

    def test_final_front_object_occludes_back_object_after_all_pose_updates(self):
        result,rows=self.check_scene(.5,False)
        self.assertEqual(rows['back']['iou'],1.)
        self.assertTrue(rows['back']['local_overlap_pass'])
        self.assertTrue(rows['back']['final_scene_occlusion_rechecked'])
        self.assertEqual(result['object_frames_checked'],2)
        self.assertFalse(result['poses_modified'])

    def test_newly_visible_mismatch_rejects_a_stale_pass(self):
        result,rows=self.check_scene(.7,True)
        self.assertAlmostEqual(rows['back']['iou'],.5)
        self.assertFalse(rows['back']['local_overlap_pass'])
        self.assertGreater(result['pass_flags_changed'],0)


if __name__=='__main__':unittest.main()
