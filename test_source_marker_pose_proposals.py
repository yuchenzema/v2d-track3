import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import cv2
import numpy as np
import trimesh
from scipy.spatial.transform import Rotation

from improvement_common import digest
from source_cad_alignment import metrics,qualifies
from source_marker_pose_proposals import bootstrap,load_reviews
from source_mask_rasterizer import MeshSilhouette
from sphere_marker_reference import project_centers
from contour_pose_refinement import propose


class ReviewedMarkerPoseTests(unittest.TestCase):
    def scene(self):
        centers=[[-.045,-.07,-.016],[.037,-.05,-.016],[-.02,.075,-.016]]
        parts=[trimesh.creation.box(extents=[.13,.20,.008])]
        template=dict(markers=[dict(center_m=p,radius_m=.006) for p in centers])
        for center in centers:
            sphere=trimesh.creation.icosphere(subdivisions=2,radius=.006);sphere.apply_translation(center);parts.append(sphere)
        mesh=trimesh.util.concatenate(parts);intr=dict(fx=700.,fy=700.,cx=512.,cy=384.)
        renderer=MeshSilhouette(mesh.vertices,mesh.faces,intr)
        truth=np.eye(4);truth[:3,:3]=Rotation.from_rotvec([.15,.10,.20]).as_matrix();truth[2,3]=.6
        target=renderer.render(truth);mask=cv2.resize(target.astype('uint8'),(1024,768),interpolation=cv2.INTER_NEAREST)>0
        image=np.full((768,1024,3),10,np.uint8);image[mask]=[60,135,190]
        uv,radii,_=project_centers(template,truth,intr)
        for xy,radius in zip(uv,radii):cv2.circle(image,tuple(np.rint(xy).astype(int)),round(radius),(150,150,150),-1)
        points=[dict(xy=xy.tolist(),radius_px=float(radius)) for xy,radius in zip(uv[::-1],radii[::-1])]
        return template,renderer,intr,truth,target,mask,image,points

    def test_source_caps_resolve_pose_without_providing_marker_ids_or_pose_labels(self):
        template,renderer,intr,truth,target,mask,image,points=self.scene()
        proposals=bootstrap(image,template,renderer,intr,mask,np.zeros(mask.shape,'uint8'),target,np.ones(target.shape,bool),points,metrics,qualifies)
        self.assertTrue(proposals)
        self.assertTrue(qualifies(proposals[0]['metrics']))
        self.assertLess(np.linalg.norm(proposals[0]['pose'][:3,3]-truth[:3,3]),.01)
        self.assertLess(Rotation.from_matrix(truth[:3,:3].T@proposals[0]['pose'][:3,:3]).magnitude(),.15)

    def test_later_reassociation_regression_cannot_erase_valid_initial_geometry(self):
        template,renderer,intr,truth,target,mask,image,points=self.scene()
        solver=cv2.solvePnPGeneric;initial_seen=False;regressions=0
        def regressing_solver(*args,**kwargs):
            nonlocal initial_seen,regressions
            result=solver(*args,**kwargs)
            if initial_seen:
                # Emulate a later cap reassociation onto a nearby feature.
                # Evaluate the resulting poses against real rendered geometry.
                moved=[t.copy()+np.array([[.03],[0.],[0.]]) for t in result[2]]
                regressions+=1
                return result[0],result[1],tuple(moved),result[3]
            for rv,tv in zip(result[1],result[2]):
                angle=Rotation.from_matrix(truth[:3,:3].T@cv2.Rodrigues(rv)[0]).magnitude()
                if angle<.15 and np.linalg.norm(tv.ravel()-truth[:3,3])<.01:
                    initial_seen=True;break
            return result
        with patch('source_marker_pose_proposals.cv2.solvePnPGeneric',side_effect=regressing_solver):
            proposals=bootstrap(image,template,renderer,intr,mask,np.zeros(mask.shape,'uint8'),target,np.ones(target.shape,bool),points,metrics,qualifies)
        self.assertTrue(initial_seen);self.assertGreater(regressions,0)
        self.assertTrue(proposals)
        self.assertTrue(all(qualifies(p['metrics']) for p in proposals))
        self.assertLess(np.linalg.norm(proposals[0]['pose'][:3,3]-truth[:3,3]),.01)

    def test_missing_image_caps_never_create_a_pose_from_review_points(self):
        template=dict(markers=[dict(center_m=p,radius_m=.006) for p in [[0,0,0],[.1,0,0],[0,.1,0]]])
        image=np.full((100,100,3),255,np.uint8);mask=np.ones((100,100),bool)
        points=[dict(xy=p,radius_px=5.) for p in [[20,20],[70,20],[20,70]]]
        self.assertEqual(bootstrap(image,template,None,{},mask,np.zeros(mask.shape,'uint8'),None,None,points,metrics,qualifies),[])

    def test_cap_pnp_outside_body_gate_is_recovered_only_with_source_contour_and_caps(self):
        template,renderer,intr,truth,target,mask,image,points=self.scene()
        solver=cv2.solvePnPGeneric
        def biased_solver(*args,**kwargs):
            result=solver(*args,**kwargs)
            return result[0],result[1],tuple(t+np.array([[.004],[0.],[0.]]) for t in result[2]),result[3]
        with patch('source_marker_pose_proposals.cv2.solvePnPGeneric',side_effect=biased_solver),patch('source_marker_pose_proposals.contour_pose_proposal',wraps=propose) as contour:
            proposals=bootstrap(image,template,renderer,intr,mask,np.zeros(mask.shape,'uint8'),target,np.ones(target.shape,bool),points,metrics,qualifies)
        self.assertGreater(contour.call_count,0)
        self.assertTrue(proposals)
        self.assertTrue(all(qualifies(p['metrics']) for p in proposals))
        self.assertTrue(all(sum(o['observed'] for o in p['observations'])>=3 for p in proposals))
        self.assertLess(np.linalg.norm(proposals[0]['pose'][:3,3]-truth[:3,3]),.01)

    def test_review_is_bound_to_original_frame_hash_and_dataset_split(self):
        with tempfile.TemporaryDirectory() as directory:
            root=Path(directory);rec=root/'reconstruction';(rec/'frames').mkdir(parents=True)
            frame=rec/'frames/000000.png';frame.write_bytes(b'original source bytes')
            cfg=dict(episode=23,dataset_split='public',frames=1,objects=['wooden_piece_2'])
            row=dict(frame=0,object='wooden_piece_2',source_frame_sha256=digest(frame),points=[dict(xy=p,radius_px=5.) for p in [[20,20],[70,20],[20,70]]])
            path=root/'public_000023.json';path.write_text(json.dumps(dict(episode=23,dataset_split='public',reviews=[row])))
            reviews,binding=load_reviews(cfg,rec,root);self.assertEqual(reviews,[row]);self.assertEqual(binding['sha256'],digest(path))
            frame.write_bytes(b'different source frame')
            with self.assertRaisesRegex(ValueError,'current original source frame'):load_reviews(cfg,rec,root)
            frame.write_bytes(b'original source bytes');data=json.loads(path.read_text());data['dataset_split']='evaluation';path.write_text(json.dumps(data))
            with self.assertRaisesRegex(ValueError,'another episode or split'):load_reviews(cfg,rec,root)
