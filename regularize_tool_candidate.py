"""Resolve planar-tool orientation ambiguity and isolated monocular depth outliers."""
from pathlib import Path
import json,sys
import numpy as np,trimesh
from scipy.ndimage import median_filter
from scipy.spatial.transform import Rotation,Slerp
from pose_continuity import optimize
r=Path(sys.argv[1]).resolve();rec=r/'reconstruction';name=sys.argv[2] if len(sys.argv)>2 else 'scene_motion_image_geometry_candidate.npz';floor_mode=name=='scene_motion_floor_candidate.npz';prefix='floor-geometry' if floor_mode else 'image-geometry'
assert not (r/'coherent-input-checksums.json').exists()
optimize(r,motion_name=name,marker_name=prefix+'-pose-continuity.json',object_names=['wooden_spoon'],max_rotation_step=.6,respect_support=floor_mode,preserve_tool_axis=floor_mode)
p=rec/name;d=np.load(p);out={k:d[k].copy() for k in d.files};cfg=json.loads((r/'episode-config.json').read_text());b=cfg['objects'].index('wooden_spoon');M=out['object_to_world'][:,b];C=out['camera_to_world'];center=trimesh.load(rec/'mesh_priors/wooden_spoon.ply',process=False).vertices.mean(0);world=np.einsum('tij,j->ti',M[:,:3,:3],center)+M[:,:3,3];local=np.einsum('tji,tj->ti',C[:,:3,:3],world-C[:,:3,3]);z=local[:,2].copy();filtered=median_filter(z,size=5,mode='nearest');delta=np.clip(filtered-z,-.06,.06)
# Keep the original projected centroid: adjust only uncertain metric depth.
local*=((z+delta)/z)[:,None];world2=np.einsum('tij,tj->ti',C[:,:3,:3],local)+C[:,:3,3];M[:,:3,3]=world2-np.einsum('tij,j->ti',M[:,:3,:3],center);lifts=np.zeros(len(M));rests=[]
if floor_mode:
 vertices=np.asarray(trimesh.load(rec/'mesh_priors/wooden_spoon.ply',process=False).vertices);radius=json.loads((rec/'scene-reconstruction.json').read_text())['table_support_radius_m']
 observed={32:[(0,35,0,20),(211,len(M)-1,252,258)],33:[(0,41,0,20),(289,len(M)-1,337,347)],35:[(0,14,0,9),(396,len(M)-1,408,420)],36:[(0,19,0,14),(379,len(M)-1,390,406)]}
 for lo,hi,a,z in observed.get(cfg['episode'],[]):
  from flat_tool_rests import fit
  ids=np.unique(np.linspace(a,z,min(7,z-a+1)).round().astype(int));model,diagnostic=fit(r,vertices,center,C,ids,b,M[a:z+1,:3,:3]);fixed_center=model[:3,:3]@center+model[:3,3]
  for i in list(range(max(0,lo-4),lo))+list(range(hi+1,min(len(M),hi+5))):
   weight=(i-lo+5)/5 if i<lo else (hi+5-i)/5;current_center=M[i,:3,:3]@center+M[i,:3,3];rotation=Slerp([0,1],Rotation.from_matrix([M[i,:3,:3],model[:3,:3]]))([weight])[0].as_matrix();M[i,:3,:3]=rotation;M[i,:3,3]=(1-weight)*current_center+weight*fixed_center-rotation@center
  M[lo:hi+1]=model;rests.append({'interval':[lo,hi],'clear_anchor_frames':[a,z],'source_observed_flat_rest':True,'boundary_blend_frames':4,'image_fit':diagnostic})
 for i,pose in enumerate(M):
  world_vertices=vertices@pose[:3,:3].T+pose[:3,3];inside=np.linalg.norm(world_vertices[:,:2],axis=1)<radius
  if inside.any():lifts[i]=max(0.,.6505-world_vertices[inside,2].min());M[i,2,3]+=lifts[i]
 world2=np.einsum('tij,j->ti',M[:,:3,:3],center)+M[:,:3,3]
out['object_to_world'][:,b]=M;np.savez_compressed(p,**out)
R=Rotation.from_matrix(M[:,:3,:3]);report={'method':__doc__,'maximum_camera_depth_adjustment_m':float(abs(delta).max()),'maximum_centroid_step_m':float(np.linalg.norm(np.diff(world2,axis=0),axis=1).max()),'maximum_orientation_step_rad':float((R[:-1].inv()*R[1:]).magnitude().max()),'source_timestamps_unchanged':True,'projected_centroid_preserved_by_depth_filter':True,'support_projection_can_change_projected_centroid':floor_mode,'maximum_final_support_lift_m':float(lifts.max()),'observed_rest_intervals':rests};(rec/(prefix+'-depth-continuity.json')).write_text(json.dumps(report,indent=2));print(json.dumps(report),flush=True)
