"""Resolve unobservable pose branches before they become policy motor targets."""
from pathlib import Path
import json
import numpy as np,trimesh
from scipy.spatial.transform import Rotation,Slerp

def optimize(root,motion_name='scene_motion_physical.npz',marker_name='pose-continuity.json',object_names=None,max_rotation_step=None,respect_support=False,preserve_tool_axis=False):
 root=Path(root);rec=root/'reconstruction';marker=rec/marker_name
 if marker.exists():return False
 cfg=json.loads((root/'episode-config.json').read_text());p=rec/motion_name;d=np.load(p);out={k:d[k].copy() for k in d.files};report={};T=len(out['object_to_world'])
 support_radius=json.loads((rec/'scene-reconstruction.json').read_text())['table_support_radius_m'] if respect_support else None
 for b,name in enumerate(cfg['objects']):
  if object_names is not None and name not in object_names:continue
  old=out['object_to_world'][:,b].copy();M=old.copy();mesh=trimesh.load(rec/f'mesh_priors/{name}.ply',process=False);center=mesh.vertices.mean(0);worldcenter=np.einsum('nij,j->ni',M[:,:3,:3],center)+M[:,:3,3];fixed=[]
  if name in ['blue_cup','beige_cup']:
   # The scans are approximately axial. Rotation about the cup axis is weakly
   # observed through occlusion and independent pose registration changes it
   # arbitrarily. Choose minimum twist, preserving measured rim-axis tilt.
   import cv2
   intr=json.loads((rec/'camera_intrinsics.json').read_text());camera=out['camera_to_world'];measure=np.load(rec/'hand_measurements.npz');vertices=mesh.vertices[::max(1,len(mesh.vertices)//30000)];flip=np.diag([1.,-1.,-1.]);axes=np.stack([old[:,:3,2],-old[:,:3,2]],axis=1);unary=np.zeros((T,2));candidates=np.repeat(old[:,None],2,axis=1);candidates[:,1,:3,:3]=old[:,:3,:3]@flip;candidates[:,1,:3,3]=worldcenter-np.einsum('nij,j->ni',candidates[:,1,:3,:3],center)
   hints=cfg.get('orientation_hints',{}).get(name,{})
   for i in range(T):
    target=cv2.resize(cv2.imread(str(rec/f'masks/{b+1}/{i:06d}.png'),0),(512,384),interpolation=cv2.INTER_NEAREST)>0;valid=np.ones((384,512),bool)
    for j in range(len(cfg['objects'])):
     if j!=b:valid&=cv2.resize(cv2.imread(str(rec/f'masks/{j+1}/{i:06d}.png'),0),(512,384),interpolation=cv2.INTER_NEAREST)==0
    for j in range(2):
     P=np.linalg.inv(camera[i])@candidates[i,j];v=vertices@P[:3,:3].T+P[:3,3];uv=np.rint((v[:,:2]/np.maximum(.001,v[:,2:])*[intr['fx'],intr['fy']]+[intr['cx'],intr['cy']])*.5).astype(int);ok=(v[:,2]>.05)&(uv[:,0]>=0)&(uv[:,0]<512)&(uv[:,1]>=0)&(uv[:,1]<384);render=np.zeros((384,512),np.uint8);render[uv[ok,1],uv[ok,0]]=255;render=cv2.morphologyEx(render,cv2.MORPH_CLOSE,np.ones((3,3),np.uint8))>0;score=(render&target&valid).sum()/max(1,((render|target)&valid).sum());unary[i,j]=1-score
     for state,spans in hints.items():
      if any(a<=i<=z for a,z in spans):unary[i,j]+=.8*max(0.,axes[i,j,2]*(1 if state=='upright' else -1))
   cost=unary.copy();back=np.zeros((T,2),int)
   for i in range(1,T):
    for j in range(2):
     angles=np.arccos(np.clip(axes[i-1]@axes[i,j],-1,1));values=cost[i-1]+.12*angles**2;k=int(values.argmin());cost[i,j]+=values[k];back[i,j]=k
   state=int(cost[-1].argmin());choices=np.zeros(T,int)
   for i in range(T-1,-1,-1):choices[i]=state;state=back[i,state]
   M=candidates[np.arange(T),choices].copy();fixed=np.flatnonzero(choices).tolist()
   for i in range(1,T):
    D=M[i-1,:3,:3].T@M[i,:3,:3];angle=np.arctan2(D[0,1]-D[1,0],D[0,0]+D[1,1]);M[i,:3,:3]=M[i,:3,:3]@Rotation.from_rotvec([0,0,angle]).as_matrix()
   method='Approximate axial symmetry prior: minimum-twist transport of the measured cup axis. Axis polarity is selected globally using every source silhouette, temporal continuity and explicitly reviewed upright/inverted intervals; gradual observed inversion is retained. Axial marker yaw is not claimed to be measured.'
  elif name in ['wooden_spoon','mini_sweeper','mini_dust_pan','pan','wooden_piece_1','wooden_piece_2','water_pitcher'] or (name=='white_pot' and np.max((Rotation.from_matrix(old[:-1,:3,:3]).inv()*Rotation.from_matrix(old[1:,:3,:3])).magnitude())>.8):
   # Planar tools can be re-registered with inconsistent surface-normal or
   # head/handle signs when occluded. Score all four rigid orientation branches
   # against visible pixels, then solve one continuous path through the video.
   import cv2
   intr=json.loads((rec/'camera_intrinsics.json').read_text());camera=out['camera_to_world'];measure=np.load(rec/'hand_measurements.npz');vertices=mesh.vertices[::max(1,len(mesh.vertices)//30000)]
   _,basis=np.linalg.eigh(np.cov(mesh.vertices.T));groups=np.array([np.eye(3),*[basis@np.diag(signs)@basis.T for signs in [[1,-1,-1],[-1,1,-1],[-1,-1,1]]]])
   if preserve_tool_axis and name=='wooden_spoon':groups=np.array([np.eye(3),np.diag([-1.,-1.,1.])])
   branches=len(groups);candidates=np.repeat(old[:,None],branches,axis=1);candidates[:,:,:3,:3]=np.einsum('tij,kjl->tkil',old[:,:3,:3],groups);candidates[:,:,:3,3]=worldcenter[:,None]-np.einsum('tkij,j->tki',candidates[:,:,:3,:3],center);unary=np.zeros((T,branches))
   for i in range(T):
    target=cv2.resize(cv2.imread(str(rec/f'masks/{b+1}/{i:06d}.png'),0),(512,384),interpolation=cv2.INTER_NEAREST)>0;occlusion=np.zeros((768,1024),np.uint8)
    for side in range(2):
     if not measure['observed'][i,side]:continue
     uv=np.rint(measure['uv'][i,side]).astype(int);cv2.fillConvexPoly(occlusion,cv2.convexHull(uv[[0,1,5,9,13,17]]),255)
     for finger in range(5):
      ids=[0]+list(range(1+4*finger,5+4*finger))
      for ia,ib in zip(ids[:-1],ids[1:]):cv2.line(occlusion,tuple(uv[ia]),tuple(uv[ib]),255,14)
    valid=cv2.resize(occlusion,(512,384),interpolation=cv2.INTER_NEAREST)==0
    for other in range(len(cfg['objects'])):
     if other!=b:valid&=cv2.resize(cv2.imread(str(rec/f'masks/{other+1}/{i:06d}.png'),0),(512,384),interpolation=cv2.INTER_NEAREST)==0
    for j in range(branches):
     P=np.linalg.inv(camera[i])@candidates[i,j];v=vertices@P[:3,:3].T+P[:3,3];uv=np.rint((v[:,:2]/np.maximum(.001,v[:,2:])*[intr['fx'],intr['fy']]+[intr['cx'],intr['cy']])*.5).astype(int);ok=(v[:,2]>.05)&(uv[:,0]>=0)&(uv[:,0]<512)&(uv[:,1]>=0)&(uv[:,1]<384);render=np.zeros((384,512),np.uint8);render[uv[ok,1],uv[ok,0]]=255;render=cv2.morphologyEx(render,cv2.MORPH_CLOSE,np.ones((3,3),np.uint8));contours,_=cv2.findContours(render,cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_SIMPLE);cv2.drawContours(render,contours,-1,255,-1);render=render>0;score=(render&target&valid).sum()/max(1,((render|target)&valid).sum());unary[i,j]=1-score+.005*(j!=0)
     if respect_support:
      world=vertices@candidates[i,j,:3,:3].T+candidates[i,j,:3,3];inside=np.linalg.norm(world[:,:2],axis=1)<support_radius
      if inside.any():unary[i,j]+=min(5.,100*max(0.,.646-world[inside,2].min()))
   hints=cfg.get('rigid_object_orientation_hints',{}).get(name,{})
   if hints:
    local_up=old[0,:3,:3].T@np.array([0.,0.,1.]);vertical=np.einsum('tkij,j->tki',candidates[:,:,:3,:3],local_up)[:,:,2]
    for state,spans in hints.items():
     for lo,hi in spans:unary[lo:hi+1]+=.8*np.maximum(0.,.5+vertical[lo:hi+1]*(1 if state=='inverted' else -1))
   cost=unary.copy();cost[0,1:]+=.5;back=np.zeros((T,branches),int)
   for i in range(1,T):
    for j in range(branches):
     rotations=np.einsum('kji,jl->kil',candidates[i-1,:,:3,:3],candidates[i,j,:3,:3]);angles=Rotation.from_matrix(rotations).magnitude();values=cost[i-1]+.20*angles**2;k=int(values.argmin());cost[i,j]+=values[k];back[i,j]=k
   state=int(cost[-1].argmin());choices=np.zeros(T,int)
   for i in range(T-1,-1,-1):choices[i]=state;state=back[i,state]
   M=candidates[np.arange(T),choices].copy();fixed=np.flatnonzero(choices).tolist();method='Four rigid principal-axis orientation branches selected globally using every visible source silhouette and rotational continuity; human and other-object occlusion excluded. Mesh geometry and measured centroid unchanged; hidden normal/head direction remains an estimate.'
   if branches==2:method='Two shaft-preserving planar normal branches selected globally from source masks and continuity. The visible spoon head and handle identities are retained; table intersections are penalized.'
  else:continue
  R=Rotation.from_matrix(M[:,:3,:3]);corrected=[]
  for i in range(1,T-1):
   neighbor=(R[i-1].inv()*R[i+1]).magnitude();middle=Slerp([0,1],Rotation.from_matrix([R[i-1].as_matrix(),R[i+1].as_matrix()]))([.5])[0]
   if neighbor<.35 and (middle.inv()*R[i]).magnitude()>.35:R[i]=middle;corrected.append(i)
  if name in ['blue_cup','beige_cup','wooden_spoon','mini_sweeper','mini_dust_pan','pan','wooden_piece_1','wooden_piece_2','water_pitcher','white_pot']:
   R=Rotation.from_matrix([R[max(0,i-1):min(T,i+2)].mean().as_matrix() for i in range(T)])
  if name in ['mini_sweeper','water_pitcher','white_pot'] and np.max((R[:-1].inv()*R[1:]).magnitude())>.8:
   R=Rotation.from_matrix([R[max(0,i-2):min(T,i+3)].mean().as_matrix() for i in range(T)])
  if (name in ['water_pitcher','mini_sweeper'] or max_rotation_step is not None) and np.max((R[:-1].inv()*R[1:]).magnitude())>(max_rotation_step or .8):
   angular_limit=max_rotation_step or .6
   for iteration in range(4):
    forward=[R[0]]
    for i in range(1,T):
     delta=(forward[-1].inv()*R[i]).as_rotvec();a=np.linalg.norm(delta);forward.append(forward[-1]*Rotation.from_rotvec(delta*min(1.,angular_limit/max(a,1e-12))))
    backward=[R[-1]]
    for i in range(T-2,-1,-1):
     delta=(backward[-1].inv()*R[i]).as_rotvec();a=np.linalg.norm(delta);backward.append(backward[-1]*Rotation.from_rotvec(delta*min(1.,angular_limit/max(a,1e-12))))
    backward=backward[::-1];R=Rotation.from_matrix([Slerp([0,1],Rotation.from_matrix([forward[i].as_matrix(),backward[i].as_matrix()]))([.5])[0].as_matrix() for i in range(T)])
   method+=f' Remaining occlusion-induced jumps regularized by symmetric forward/backward geodesic transport with {angular_limit} rad/frame angular increments; source timing is unchanged.'
  M[:,:3,:3]=R.as_matrix();M[:,:3,3]=worldcenter-np.einsum('nij,j->ni',M[:,:3,:3],center);out['object_to_world'][:,b]=M
  oldR=Rotation.from_matrix(old[:,:3,:3]);before=(oldR[:-1].inv()*oldR[1:]).magnitude();after=(R[:-1].inv()*R[1:]).magnitude();report[name]={'method':method,'normal_branch_corrections':fixed,'isolated_outlier_frames':corrected,'rotation_step_rad_before_p50_p95_max':np.quantile(before,[.5,.95,1]).tolist(),'rotation_step_rad_after_p50_p95_max':np.quantile(after,[.5,.95,1]).tolist(),'mesh_centroid_preserved_each_frame':True}
 if report:
  backup='scene_motion_before_pose_continuity.npz' if motion_name=='scene_motion_physical.npz' else p.stem+'_before_continuity.npz'
  p.rename(rec/backup);np.savez_compressed(p,**out)
 marker.write_text(json.dumps({'objects':report,'evaluation_ground_truth_used':False},indent=2));print('Pose continuity',cfg['sequence'],json.dumps(report),flush=True);return bool(report)
if __name__=='__main__':
 import sys
 optimize(sys.argv[1])
