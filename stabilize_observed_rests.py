"""Use explicitly video-reviewed resting intervals to reject occlusion-induced object drift."""
from pathlib import Path
import json,numpy as np,trimesh
from scipy.spatial.transform import Rotation,Slerp

def optimize(root):
 r=Path(root);rec=r/'reconstruction';cfg=json.loads((r/'episode-config.json').read_text());intervals=cfg.get('observed_static_intervals',[])
 if not intervals or (rec/'observed-rest-stability.json').exists():return
 assert not (r/'coherent-input-checksums.json').exists()
 p=rec/'scene_motion_physical.npz';d=np.load(p);out={k:d[k].copy() for k in d.files};T=len(out['object_to_world']);reports=[]
 for item in intervals:
  b=item['object'];start,end=item['interval'];a,z=item['clear_anchor_frames'];original=out['object_to_world'][:,b].copy();mesh=trimesh.load(rec/f"mesh_priors/{cfg['objects'][b]}.ply",process=False);center=mesh.vertices.mean(0);anchors=original[a:z+1];worldcenter=np.einsum('tij,j->ti',anchors[:,:3,:3],center)+anchors[:,:3,3];orientation=Rotation.from_matrix(anchors[:,:3,:3]).mean();rotation=orientation.as_matrix();position=np.median(worldcenter,axis=0)-rotation@center
  for i in range(max(0,start-4),min(T,end+5)):
   blend=min(1.,(i-(start-5))/5,((end+5)-i)/5);blend=max(0.,blend)
   out['object_to_world'][i,b,:3,:3]=Slerp([0,1],Rotation.from_matrix([original[i,:3,:3],rotation]))([blend])[0].as_matrix();out['object_to_world'][i,b,:3,3]=(1-blend)*original[i,:3,3]+blend*position
  reports.append({**item,'object_name':cfg['objects'][b],'position_correction_max_m':float(np.linalg.norm(out['object_to_world'][:,b,:3,3]-original[:,:3,3],axis=1).max()),'method':'Robust stationary world pose from clear source anchor frames; gradual four-frame blend into adjacent observed motion. Rest intervals were visually identified, not inferred solely from absence of a grasp.'})
 if (rec/'rebuilt_scene.glb').exists():
  scene=trimesh.load(rec/'rebuilt_scene.glb')
  for b,name in enumerate(cfg['objects']):scene.graph.update(frame_to=name,matrix=out['object_to_world'][0,b])
  scene.export(rec/'rebuilt_scene.glb')
 p.rename(rec/'scene_motion_before_observed_rest_stability.npz');np.savez_compressed(p,**out);(rec/'observed-rest-stability.json').write_text(json.dumps({'method':__doc__,'intervals':reports,'evaluation_ground_truth_used':False},indent=2));print('Observed-rest stabilization',cfg['sequence'],json.dumps(reports),flush=True)
if __name__=='__main__':
 import sys
 optimize(sys.argv[1])
