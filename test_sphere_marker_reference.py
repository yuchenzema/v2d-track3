import unittest
import cv2
import numpy as np
import trimesh
from unittest.mock import patch
from sphere_marker_reference import curvature_spheres,measure_markers,marker_residual,circle_candidates,edge_circle_candidates


class SphereReferenceTests(unittest.TestCase):
    def test_grey_cap_radial_edges_are_measured_independently_of_blob_separation(self):
        image=np.full((100,100,3),220,np.uint8)
        cv2.circle(image,(50,45),7,(150,150,150),-1)
        cv2.rectangle(image,(48,50),(52,74),(130,130,130),-1)
        candidates=edge_circle_candidates(image,np.array([51.,46.]),7.)
        self.assertTrue(candidates)
        best=min(candidates,key=lambda row:row['cost'])
        self.assertLess(np.linalg.norm(np.array(best['xy'])-[50,45]),1.5)
        self.assertGreaterEqual(best['edge_arc_coverage'],.7)
        self.assertEqual(best['hough_thresholds'],[10,12])

    def test_radial_search_rejects_coloured_circles_and_straight_edges(self):
        colored=np.full((100,100,3),220,np.uint8)
        cv2.circle(colored,(50,45),7,(30,70,170),-1)
        self.assertEqual(edge_circle_candidates(colored,np.array([50.,45.]),7.),[])
        straight=np.full((100,100,3),220,np.uint8)
        cv2.rectangle(straight,(47,25),(53,75),(150,150,150),-1)
        self.assertEqual(edge_circle_candidates(straight,np.array([50.,45.]),7.),[])

    def test_hough_vote_without_radial_image_evidence_never_creates_a_cap(self):
        fake=np.array([[[24.5,24.5,6.]]],np.float32)
        with patch('sphere_marker_reference.cv2.HoughCircles',return_value=fake):
            self.assertEqual(edge_circle_candidates(np.full((100,100,3),150,np.uint8),np.array([50.,50.]),6.),[])

    def test_bright_grey_cap_survives_only_the_narrow_threshold_interval(self):
        image=np.full((100,100,3),220,np.uint8)
        cv2.circle(image,(50,45),6,(185,185,185),-1)
        cv2.line(image,(50,50),(50,68),(60,60,60),2)
        candidates=circle_candidates(image,np.array([50.,45.]),6.)
        self.assertTrue(candidates)
        best=min(candidates,key=lambda row:row['cost'])
        self.assertLess(np.linalg.norm(np.array(best['xy'])-[50,45]),.8)
        self.assertGreaterEqual(len(best['intensity_levels']),2)
        self.assertIn(208,best['intensity_levels'])

    def test_rejected_distant_circle_does_not_suppress_the_real_cap(self):
        image=np.full((100,100,3),255,np.uint8)
        cv2.rectangle(image,(25,20),(61,70),(220,220,220),-1)
        cv2.circle(image,(50,45),6,(150,150,150),-1)
        cv2.line(image,(50,50),(50,68),(60,60,60),2)
        cv2.circle(image,(71,45),4,(90,90,90),-1)
        candidates=circle_candidates(image,np.array([50.,45.]),6.)
        self.assertTrue(candidates)
        best=min(candidates,key=lambda row:row['cost'])
        self.assertLess(np.linalg.norm(np.array(best['xy'])-[50,45]),.8)
        self.assertLess(best['cost'],4.5)
        self.assertEqual(best['method'],'measured_multithreshold_grey_cap')
        self.assertTrue(any(row['method']=='measured_low_saturation_circular_component' and row['cost']>=4.5 for row in candidates))

    def test_grey_cap_on_white_plastic_is_measured_despite_connected_stem(self):
        image=np.full((100,100,3),220,np.uint8)
        cv2.circle(image,(50,45),6,(150,150,150),-1)
        cv2.line(image,(50,50),(50,68),(60,60,60),2)
        candidates=circle_candidates(image,np.array([52.,47.]),6.)
        self.assertTrue(candidates)
        best=min(candidates,key=lambda row:row['cost'])
        self.assertLess(np.linalg.norm(np.array(best['xy'])-[50,45]),.8)
        self.assertEqual(best['method'],'measured_multithreshold_grey_cap')
        self.assertGreaterEqual(len(best['intensity_levels']),2)

    def test_white_plastic_without_a_cap_has_no_invented_circle(self):
        candidates=circle_candidates(np.full((100,100,3),220,np.uint8),np.array([50.,50.]),6.)
        self.assertEqual(candidates,[])

    def test_narrow_dark_line_does_not_become_a_measured_marker(self):
        image=np.full((100,100,3),220,np.uint8);cv2.rectangle(image,(48,34),(51,66),(150,150,150),-1)
        self.assertEqual(circle_candidates(image,np.array([50.,50.]),6.),[])

    def test_back_marker_hidden_by_front_cad_surface_is_not_an_image_measurement(self):
        template={'markers':[{'center_m':[-.02,0.,0.],'radius_m':.01},{'center_m':[.02,0.,.2],'radius_m':.01}]}
        pose=np.eye(4);pose[2,3]=1.;intr={'fx':500.,'fy':500.,'cx':100.,'cy':100.}
        surface=np.full((100,100),.99,np.float32)
        def candidates(image,prediction,radius):return [dict(xy=prediction.tolist(),radius_px=radius,circularity=1.,cost=0.)]
        with patch('sphere_marker_reference.circle_candidates',side_effect=candidates):
            rows=measure_markers(np.zeros((200,200,3),np.uint8),template,pose,intr,surface_depth=surface)
        self.assertTrue(rows[0]['observed']);self.assertFalse(rows[1]['observed'])
        self.assertEqual(rows[1]['visibility'],'self_occluded');self.assertNotIn('measurement',rows[1])

    def test_sphere_centres_are_recovered_in_metric_object_frame(self):
        sphere=trimesh.creation.icosphere(subdivisions=3,radius=.007)
        center=np.array([.02,-.03,.01]);sphere.apply_translation(center)
        found=curvature_spheres(sphere.vertices,sphere.faces)
        self.assertTrue(found)
        self.assertLess(np.linalg.norm(np.array(found[0]['center_m'])-center),.0002)
        self.assertAlmostEqual(found[0]['radius_m'],.007,delta=.0002)

    def test_a_missing_image_marker_is_not_replaced_by_the_model_prediction(self):
        template={'markers':[{'center_m':[0.,0.,0.],'radius_m':.007}]}
        pose=np.eye(4);pose[2,3]=.7;intr={'fx':400.,'fy':400.,'cx':50.,'cy':50.}
        observations=measure_markers(np.zeros((100,100,3),np.uint8),template,pose,intr)
        self.assertFalse(observations[0]['observed']);self.assertNotIn('measurement',observations[0])

    def test_reprojection_uses_scan_coordinates_and_camera_intrinsics(self):
        pose=np.eye(4);pose[2,3]=1.;intr={'fx':400.,'fy':400.,'cx':50.,'cy':50.}
        residual=marker_residual(pose,[[.01,0,0]],[[54.,50.]],intr)
        np.testing.assert_allclose(residual,0.,atol=1e-9)
