import unittest
from unittest.mock import patch

import numpy as np
from PIL import Image
import trimesh
from scipy.spatial.transform import Rotation

from source_cad_alignment import prepare_budget_mesh
from source_mask_rasterizer import MeshSilhouette


def seamed_surface():
    original=trimesh.creation.icosphere(subdivisions=3,radius=.06)
    original.apply_scale([1.,.8,1.2])
    vertices=original.vertices[original.faces].reshape(-1,3)
    result=trimesh.Trimesh(vertices=vertices,faces=np.arange(len(vertices)).reshape(-1,3),process=False)
    uv=np.c_[np.linspace(0,1,len(vertices)),np.arange(len(vertices))%3/2]
    result.visual=trimesh.visual.texture.TextureVisuals(uv=uv,image=Image.new('RGB',(2,2),'white'))
    return result


class SourceMeshBudgetTests(unittest.TestCase):
    def test_textured_scan_seams_do_not_become_holes_after_decimation(self):
        original=seamed_surface();simplified=prepare_budget_mesh(original,256)
        self.assertLessEqual(len(simplified.faces),256)
        self.assertTrue(simplified.is_watertight)
        intr=dict(fx=600.,fy=600.,cx=512.,cy=384.)
        before=MeshSilhouette(original.vertices,original.faces,intr)
        after=MeshSilhouette(simplified.vertices,simplified.faces,intr)
        for angles in [[.8,.4,.2],[1.2,-.3,.5],[.3,.7,-.2]]:
            with self.subTest(angles=angles):
                pose=np.eye(4);pose[:3,:3]=Rotation.from_euler('xyz',angles).as_matrix();pose[:3,3]=[.02,-.01,.65]
                a,b=before.render(pose),after.render(pose)
                self.assertGreater((a&b).sum()/max(1,(a|b).sum()),.95)

    def test_source_positions_triangles_and_texture_coordinates_are_immutable(self):
        original=seamed_surface();vertices=original.vertices.copy();faces=original.faces.copy();uv=original.visual.uv.copy()
        prepare_budget_mesh(original,256)
        np.testing.assert_array_equal(original.vertices,vertices)
        np.testing.assert_array_equal(original.faces,faces)
        np.testing.assert_array_equal(original.visual.uv,uv)

    def test_mesh_already_under_budget_keeps_its_metric_geometry(self):
        original=trimesh.creation.box(extents=[.02,.05,.13]);original.apply_translation([.01,-.02,.07])
        result=prepare_budget_mesh(original,128)
        np.testing.assert_array_equal(result.vertices,original.vertices)
        np.testing.assert_array_equal(result.faces,original.faces)

    def test_decimator_cannot_silently_export_an_oversized_mesh(self):
        original=seamed_surface()
        with patch.object(trimesh.Trimesh,'simplify_quadric_decimation',return_value=original):
            with self.assertRaisesRegex(RuntimeError,'exceeded'):
                prepare_budget_mesh(original,128)
