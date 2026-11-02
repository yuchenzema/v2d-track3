import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import cv2
import numpy as np

import fresh_evaluation_frontend as frontend


class SourceFrameCacheRestoreTests(unittest.TestCase):
    def setUp(self):
        self.tmp=tempfile.TemporaryDirectory();self.addCleanup(self.tmp.cleanup)
        self.base=Path(self.tmp.name);self.root=self.base/'run';self.rec=self.root/'reconstruction'
        self.frames=self.rec/'frames';self.frames.mkdir(parents=True)
        self.video=self.base/'source.mp4';writer=cv2.VideoWriter(str(self.video),cv2.VideoWriter_fourcc(*'mp4v'),20.,(64,48))
        self.assertTrue(writer.isOpened())
        for frame in range(4):
            image=np.full((48,64,3),frame*45,np.uint8);image[10:30,20:40]=(20,90,180);writer.write(image)
        writer.release()
        K=np.array([[45.,0.,30.],[0.,46.,22.],[0.,0.,1.]])
        D=np.array([.03,-.01,0.,0.,0.,0.,0.,0.,0.,0.,0.,0.,0.,0.])
        self.calib=self.base/'datasets/video_to_data_challenge/track_3/public/meta/camera_calibration.json'
        self.calib.parent.mkdir(parents=True)
        self.write(self.calib,dict(cameras=dict(ego_cam_a=dict(resolution=[64,48],intrinsic_matrix=K.tolist(),distortion_coefficients=D.tolist()))))
        new_K,_=cv2.getOptimalNewCameraMatrix(K,D,(64,48),.7,(64,48));mx,my=cv2.initUndistortRectifyMap(K,D,None,new_K,(64,48),cv2.CV_32FC1)
        self.expected=[];cap=cv2.VideoCapture(str(self.video))
        for frame in range(4):
            ok,image=cap.read();self.assertTrue(ok)
            self.expected.append(cv2.rotate(cv2.remap(cv2.rotate(image,cv2.ROTATE_180),mx,my,cv2.INTER_LINEAR),cv2.ROTATE_180))
        cap.release()
        self.intr=dict(fx=new_K[0,0],fy=new_K[1,1],cx=63-new_K[0,2],cy=47-new_K[1,2],width=64,height=48)
        self.write(self.rec/'camera_intrinsics.json',self.intr)
        self.write(self.rec/'video-preparation.json',dict(input_video=str(self.video),frames=4,resolution=[64,48],
            calibration_source='public/meta/camera_calibration.json',undistortion_alpha=.7))
        self.cfg=dict(source=str(self.video),frames=4,fps=20.)
        self.measured=self.rec/'hand_measurements.npz';self.measured.write_bytes(b'preserved source hand observations')

    @staticmethod
    def write(path,data):path.write_text(json.dumps(data))

    def restore(self):
        with patch.object(frontend,'BASE',self.base):return frontend.restore_source_frame_cache(self.root,self.cfg)

    def test_preparation_marker_with_only_frame_zero_restores_every_missing_original_frame(self):
        cv2.imwrite(str(self.frames/'000000.png'),self.expected[0]);before=(self.frames/'000000.png').read_bytes()
        report=self.restore()
        self.assertEqual(report['restored_frames'],[1,2,3]);self.assertEqual(report['verified_source_frames'],4)
        for frame,image in enumerate(self.expected):np.testing.assert_array_equal(cv2.imread(str(self.frames/f'{frame:06d}.png')),image)
        self.assertEqual((self.frames/'000000.png').read_bytes(),before)
        self.assertEqual(self.measured.read_bytes(),b'preserved source hand observations')
        self.assertFalse(report['evaluation_ground_truth_used']);self.assertIn('source_video_sha256',report)

    def test_truncated_png_is_detected_and_replaced_from_original_source(self):
        (self.frames/'000002.png').write_bytes(b'truncated png')
        self.restore();np.testing.assert_array_equal(cv2.imread(str(self.frames/'000002.png')),self.expected[2])

    def test_complete_cache_is_verified_without_redecoding_original_video(self):
        for frame,image in enumerate(self.expected):cv2.imwrite(str(self.frames/f'{frame:06d}.png'),image)
        with patch.object(frontend.cv2,'VideoCapture',side_effect=AssertionError('Complete source cache needs no decode')):
            report=self.restore()
        self.assertEqual(report['restored_frames'],[]);self.assertFalse(report['original_video_used_for_restoration'])

    def test_changed_camera_intrinsics_fail_before_creating_any_frames(self):
        self.write(self.rec/'camera_intrinsics.json',{**self.intr,'cx':self.intr['cx']+1})
        with self.assertRaisesRegex(RuntimeError,'camera intrinsics'):self.restore()
        self.assertEqual(list(self.frames.iterdir()),[]);self.assertFalse((self.rec/'source-frame-cache.json').exists())

    def test_wrong_video_timing_fails_without_a_completion_marker(self):
        self.cfg['fps']=30.
        with self.assertRaisesRegex(RuntimeError,'timing'):self.restore()
        self.assertFalse((self.rec/'source-frame-cache.json').exists())

    def test_preparation_for_another_source_is_rejected(self):
        prep=json.loads((self.rec/'video-preparation.json').read_text());prep['input_video']=str(self.base/'other.mp4')
        self.write(self.rec/'video-preparation.json',prep)
        with self.assertRaisesRegex(RuntimeError,'not bound'):self.restore()
        self.assertEqual(list(self.frames.iterdir()),[])

    def test_failed_png_write_preserves_existing_file_and_does_not_certify_cache(self):
        broken=self.frames/'000000.png';broken.write_bytes(b'previous truncated file')
        with patch.object(frontend.cv2,'imwrite',return_value=False):
            with self.assertRaisesRegex(RuntimeError,'Cannot restore'):self.restore()
        self.assertEqual(broken.read_bytes(),b'previous truncated file')
        self.assertFalse((self.rec/'source-frame-cache.json').exists())
        self.assertEqual(list(self.frames.glob('*.restore-*.png')),[])


if __name__=='__main__':unittest.main()
