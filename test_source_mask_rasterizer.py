import unittest
import numpy as np
from source_mask_rasterizer import MeshSilhouette

class RasterizerTests(unittest.TestCase):
    def test_overlapping_faces_form_union_without_holes(self):
        v=np.array([[-.5,-.5,1],[.5,-.5,1],[.5,.5,1],[-.5,.5,1]])
        f=np.array([[0,1,2],[0,2,3]])
        k={'fx':40,'fy':40,'cx':64,'cy':64}
        a=MeshSilhouette(v,f,k).render(np.eye(4));b=MeshSilhouette(v,np.vstack([f,f[:,::-1]]),k).render(np.eye(4))
        np.testing.assert_array_equal(a,b);self.assertTrue(a[22:42,22:42].all());self.assertFalse(a[0,0])
    def test_geometry_behind_camera_is_not_rendered(self):
        r=MeshSilhouette(np.array([[0,0,-1],[1,0,-1],[0,1,-1]]),[[0,1,2]],{'fx':40,'fy':40,'cx':64,'cy':64})
        self.assertFalse(r.render(np.eye(4)).any())

if __name__=='__main__':unittest.main()
