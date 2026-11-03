import hashlib,json,tempfile,unittest
from pathlib import Path
import numpy as np,trimesh
from own_geometry import validate,REVISION
from own_source_quality import fuse
from source_mask_rasterizer import MeshSilhouette
from improvement_common import digest


class OwnSourceQualityTests(unittest.TestCase):
    def test_multiview_fusion_keeps_metric_body_coordinates_and_nonempty_surface(self):
        mesh=trimesh.creation.box(extents=[.1,.1,.04]);mesh.visual.vertex_colors=[160,170,180,255]
        intr=dict(fx=500.,fy=500.,cx=512.,cy=384.);renderer=MeshSilhouette(mesh.vertices,mesh.faces,intr);views=[]
        for distance in [.5,.55]:
            pose=np.eye(4);pose[2,3]=distance;mask=renderer.render(pose);depth=renderer.depth(pose)
            views.append((pose,mask,depth,depth>0,np.full((384,512,3),170,np.uint8)))
        result=fuse(mesh,views,intr)
        self.assertTrue(np.isfinite(result.vertices).all());self.assertLessEqual(len(result.faces),3600);self.assertGreater(len(result.faces),32)
        self.assertLess(np.max(np.abs(result.bounds-mesh.bounds)),.015)

    def test_iterative_pose_parent_must_itself_be_own_reconstruction(self):
        with tempfile.TemporaryDirectory() as directory:
            base=Path(directory);parent=base/'parent';child=base/'child';raw=base/'rgb-depth-input';raw.write_bytes(b'evaluation RGB and metric depth')
            for root in [parent,child]:
                (root/'research').mkdir(parents=True);assets=root/'reconstruction/mesh_priors';assets.mkdir(parents=True);(assets/'object.glb').write_bytes(b'own measured geometry')
                (root/'episode-config.json').write_text(json.dumps(dict(dataset_split='evaluation',own_geometry_required=True,objects=['object'])))
            proof=dict(revision=REVISION,public_object_meshes_used=False,public_object_textures_used=False,public_marker_templates_used=False,old_object_poses_used=False,
                objects=dict(object=dict(inputs={str(raw):digest(raw)},mesh_sha256=dict(glb=digest(parent/'reconstruction/mesh_priors/object.glb')))))
            p=parent/'research/own-geometry-provenance.json';p.write_text(json.dumps(proof))
            derived=dict(proof,revision='evaluation_rgb_depth_multiview_v4_20261103',old_object_poses_used=True,historical_scan_based_poses_used=False,own_parent_run=str(parent),own_parent_provenance_sha256=digest(p))
            c=child/'research/own-geometry-provenance.json';c.write_text(json.dumps(derived));validate(child)
            proof['public_object_meshes_used']=True;p.write_text(json.dumps(proof));derived['own_parent_provenance_sha256']=digest(p);c.write_text(json.dumps(derived))
            with self.assertRaisesRegex(ValueError,'Forbidden'):validate(child)
