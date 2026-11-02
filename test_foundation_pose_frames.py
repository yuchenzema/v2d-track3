import ast
from contextlib import nullcontext
import importlib.util
from pathlib import Path
from types import SimpleNamespace
import unittest
import numpy as np

LIB=Path(__file__).resolve().parent/'video_to_data/reconstruction/modules/v2d_foundation_pose/lib'
spec=importlib.util.spec_from_file_location('pose_frames',LIB/'pose_frames.py')
module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module)


class FoundationPoseFrameTests(unittest.TestCase):
    def test_centered_and_original_vertices_project_to_identical_camera_points(self):
        vertices=np.array([[.02,.03,.10],[-.04,.06,-.01],[.08,-.02,.02]])
        center=np.array([.01157,.00078,.06442])
        pose=np.eye(4);pose[:3,:3]=[[0.,0.,1.],[1.,0.,0.],[0.,1.,0.]];pose[:3,3]=[.04,.11,.54]
        before=pose.copy();converted=module.pose_for_centered_mesh(pose,center)
        reference=vertices@pose[:3,:3].T+pose[:3,3]
        rendered=(vertices-center)@converted[:3,:3].T+converted[:3,3]
        np.testing.assert_allclose(rendered,reference,atol=1e-12)
        np.testing.assert_array_equal(pose,before)
        wrong=(vertices-center)@pose[:3,:3].T+pose[:3,3]
        self.assertGreater(np.linalg.norm(wrong-reference,axis=1).min(),.06)

    def test_zero_center_does_not_change_pose(self):
        pose=np.eye(4);pose[:3,3]=[.2,-.1,.7]
        np.testing.assert_array_equal(module.pose_for_centered_mesh(pose,np.zeros(3)),pose)

    def test_actual_pose_scoring_methods_render_in_centered_coordinates(self):
        tree=ast.parse((LIB/'foundation_pose_tracker.py').read_text())
        cls=next(node for node in tree.body if isinstance(node,ast.ClassDef) and node.name=='FoundationPoseTracker')
        pose=np.eye(4);pose[:3,3]=[.04,.11,.54];center=np.array([.01157,.00078,.06442]);captured=[]
        class Depth:
            def __getitem__(self,index):return self
            def cpu(self):return self
            def numpy(self):return np.ones((4,4))
        def render(k,h,w,poses,**kwargs):
            captured.append(poses.copy());return None,Depth(),None
        namespace=dict(np=np,torch=SimpleNamespace(as_tensor=lambda x,**kwargs:np.asarray(x),float=np.float32,no_grad=nullcontext),
                       pose_for_centered_mesh=module.pose_for_centered_mesh,nvdiffrast_render=render,
                       Mask=object,CameraIntrinsics=object,Transform3d=object,DepthImage=object)
        tracker=SimpleNamespace(_est=SimpleNamespace(model_center=center,mesh_tensors={}),_glctx=None)
        mask=SimpleNamespace(mask=np.ones((4,4)));intrinsics=SimpleNamespace(to_matrix=lambda:np.eye(3),height=4,width=4);transform=SimpleNamespace(to_matrix=lambda:pose)
        for name in ['_mask_iou','_score_scale']:
            with self.subTest(method=name):
                method=next(node for node in cls.body if isinstance(node,ast.FunctionDef) and node.name==name)
                exec(compile(ast.Module(body=[method],type_ignores=[]),str(LIB/'foundation_pose_tracker.py'),'exec'),namespace)
                if name=='_mask_iou':score=namespace[name](tracker,mask,intrinsics,transform);expected=1.
                else:score=namespace[name](tracker,SimpleNamespace(depth=np.ones((4,4))),mask,intrinsics,transform);expected=2.
                self.assertAlmostEqual(score,expected,places=6)
                np.testing.assert_allclose(captured[-1][0,:3,3],pose[:3,3]+center,atol=1e-12)


if __name__=='__main__':unittest.main()
