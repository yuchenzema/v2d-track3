import unittest
from pathlib import Path
import numpy as np
from robust_table_camera_template import propagate_center,adapt,refine_circle,table_component


class TableCameraFallbackTests(unittest.TestCase):
    def test_partial_arc_cannot_produce_unbounded_radius(self):
        # The old unconstrained fit drives an almost straight edge to infinity.
        x=np.linspace(-.045,.045,30);points=np.c_[x,.00001*x*x]
        result=refine_circle(points,[0.,.25],.25)
        self.assertGreaterEqual(result[2],.18)
        self.assertLessEqual(result[2],.36)
        self.assertLessEqual(np.max(np.abs(result[:2]-[0.,.25])),.040001)

    def test_measured_circle_is_recovered_without_changing_scale(self):
        a=np.linspace(-2.,2.,70);center=np.array([.1,-.08]);radius=.25
        points=center+radius*np.c_[np.cos(a),np.sin(a)]
        result=refine_circle(points,center+[.01,-.01],.26)
        np.testing.assert_allclose(result,np.r_[center,radius],atol=1e-5)

    def test_detached_skin_component_does_not_enter_table_fit(self):
        mask=np.zeros((100,100),bool);mask[20:80,40:90]=True;mask[10:20,5:15]=True
        clean=table_component(mask)
        self.assertFalse(clean[15,10]);self.assertTrue(clean[40,60])

    def test_metric_camera_motion_preserves_static_world_table_center(self):
        previous=np.eye(4);current=np.eye(4);current[:3,3]=[.1,-.03,.1]
        result=propagate_center([0.,0.,.7],previous,current)
        np.testing.assert_allclose(current[:3,:3]@result+current[:3,3],[0.,0.,.7])

    def test_measured_plane_correction_only_changes_plane_normal_component(self):
        result=propagate_center([.1,.2,.7],np.eye(4),np.eye(4),[0.,0.,1.],.65)
        np.testing.assert_allclose(result,[.1,.2,.65])

    def test_non_finite_motion_is_rejected(self):
        current=np.eye(4);current[0,3]=np.nan
        with self.assertRaises(ValueError):propagate_center([0,0,1],np.eye(4),current)

    def test_generated_template_compiles_and_marks_missing_circle_measurements(self):
        template=Path(__file__).parent/'runs/episode_000005_newton_20261015/scripts/calibrate_table_camera.py'
        result=adapt(template.read_text());compile(result,'calibration-template','exec')
        self.assertIn('circle_fallback_frames',result)
        self.assertIn('Initial table anchor is not measurable',result)


if __name__=='__main__':unittest.main()
