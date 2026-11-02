"""Add time-resolved tool evidence and stable observed release support to a cold reconstruction."""
import argparse,json,subprocess,shutil
from pathlib import Path
import cv2,numpy as np,trimesh
from scipy.ndimage import distance_transform_edt,map_coordinates
from scipy.optimize import least_squares
from scipy.spatial.transform import Rotation
from improvement_common import save,utc
p=argparse.ArgumentParser();p.add_argument('--run',type=Path,required=True);a=p.parse_args();r=a.run;cfg=json.loads((r/'episode-config.json').read_text());rec=r/'reconstruction';marker=r/'research/rebuilt-source-constraints.json'
subprocess.run([str(r/'.venv-video/bin/python'),str(r.parents[1]/'repair_anatomical_hands_20261022.py'),'--run',str(r)],check=True)
if cfg['episode']==30:subprocess.run([str(r/'.venv-video/bin/python'),str(r.parents[1]/'rebuild_rack_geometry_20261022.py'),'--run',str(r)],check=True)
if cfg['episode']==30:subprocess.run([str(r/'.venv-video/bin/python'),str(r.parents[1]/'rebuild_brush_geometry_20261023.py'),'--run',str(r)],check=True)
if marker.exists():raise SystemExit(0)
m=np.load(rec/'scene_motion_physical.npz');out={k:m[k].copy() for k in m.files};T=cfg['frames'];names=cfg['objects'];hands=np.load(rec/'hand_measurements.npz');intr=json.loads((rec/'camera_intrinsics.json').read_text());records=[]
shutil.copy2(rec/'scene_motion_physical.npz',rec/'scene_motion_before_source_constraints.npz')
if 'wooden_spoon' in names:
 source=r.parents[1]/'runs/episode_000005_newton_20261015/scripts/track_spoon_geometry.py';script=source.read_text().replace('T=288',f'T={T}');spoon=names.index('wooden_spoon');g=next(g for g in cfg['grasps'] if g['object']==spoon)
 support=cfg.get('source_support_intervals',{}).get('wooden_spoon',[[0,g['start']-1],[g['end'],T-1]])
 script=script.replace('supported=i<=43 or i>=231',f"supported=any(first<=i<=last for first,last in {support!r})")
 script=script.replace('masks/2/',f'masks/{spoon+1}/');target=r/'scripts/rebuild_track_spoon_geometry.py';target.write_text(script)
 subprocess.run([str(r/'.venv-video/bin/python'),str(target)],check=True)
 for i in range(T):
  d=json.loads((rec/f'object_poses/wooden_spoon_geometric/{i:06d}.json').read_text());M=np.eye(4);M[:3,:3]=Rotation.from_quat(d['rotation'],scalar_first=True).as_matrix();M[:3,3]=d['translation'];out['object_to_world'][i,spoon]=out['camera_to_world'][i]@M
 records.append({'object':'wooden_spoon','method':'Fresh per-frame head/end landmarks, rigid metric length and stereo depth; source pauses retained','frames':T})
if 'mini_sweeper' in names:
 b=names.index('mini_sweeper');mesh=trimesh.load(rec/'mesh_priors/mini_sweeper.ply',process=False);vertices=np.asarray(mesh.vertices);v=vertices[np.linspace(0,len(vertices)-1,min(1800,len(vertices))).astype(int)];axis=int(np.argmax(np.ptp(vertices,axis=0)));head=vertices[vertices[:,axis]<=np.percentile(vertices[:,axis],22)].mean(0);handle=vertices[vertices[:,axis]>=np.percentile(vertices[:,axis],88)].mean(0);control=np.array([head,handle]);g=next(g for g in cfg['grasps'] if g['object']==b);side=['left','right'].index(g['side']);previous=None;details=[]
 def project(p):return p[:,:2]/np.maximum(p[:,2:],.02)*[intr['fx'],intr['fy']]+[intr['cx'],intr['cy']]
 for i in range(T):
  if not g['start']<=i<=g['end']:
   details.append({'frame':i,'unheld_source_support_pose_preserved':True});continue
  mask=cv2.imread(str(rec/f'masks/{b+1}/{i:06d}.png'),0)
  assert mask is not None,('Missing source mask',i)
  if np.count_nonzero(mask)<8:
   details.append({'frame':i,'insufficient_visible_tool_pixels':True,'fresh_foundationpose_retained':True});continue
  small=cv2.resize(mask,(512,384),interpolation=cv2.INTER_NEAREST)>0;distance=distance_transform_edt(~small);thick=cv2.distanceTransform(mask,cv2.DIST_L2,5);yy,xx=np.where(thick>=max(1.,thick.max()*.7));target_head=np.array([xx.mean(),yy.mean()]);yy,xx=np.where(mask>0);points=np.c_[xx,yy];radius=np.linalg.norm(points-target_head,axis=1);target_handle=np.median(points[radius>=np.percentile(radius,98)],axis=0)
  excluded=np.zeros((384,512),np.uint8)
  for hand in range(2):
   if not hands['observed'][i,hand]:continue
   uv=np.rint(hands['uv'][i,hand]*.5).astype(np.int32);cv2.fillConvexPoly(excluded,cv2.convexHull(uv[[0,1,5,9,13,17]]),255)
   for finger in range(5):
    ids=[0]+list(range(1+4*finger,5+4*finger))
    for first,last in zip(ids[:-1],ids[1:]):cv2.line(excluded,tuple(uv[first]),tuple(uv[last]),255,7)
  valid=excluded==0
  for other in range(len(names)):
   if other!=b:valid&=cv2.resize(cv2.imread(str(rec/f'masks/{other+1}/{i:06d}.png'),0),(512,384),interpolation=cv2.INTER_NEAREST)==0
  if np.count_nonzero(small&valid)<30:
   details.append({'frame':i,'source_too_occluded_for_extra_fit':True});continue
  held=g['start']<=i<=g['end'] and hands['observed'][i,side]
  if held:target_handle=np.mean(hands['uv'][i,side,[4,8]],axis=0)
  targets=np.array([target_head,target_handle]);C=np.linalg.inv(out['camera_to_world'][i]);seed=out['object_to_world'][i,b].copy();oldrot=Rotation.from_matrix(seed[:3,:3])
  def transform(x):
   M=seed.copy();M[:3,3]+=x[:3];M[:3,:3]=(Rotation.from_rotvec(x[3:])*oldrot).as_matrix();return M
  def residual(x):
   M=transform(x);pc=(v@M[:3,:3].T+M[:3,3])@C[:3,:3].T+C[:3,3];uv=project(pc)*.5;clipped=np.clip(uv,[0,0],[511,383]);visibility=map_coordinates(valid.astype(float),clipped[:,::-1].T,order=0,mode='nearest');silhouette=(map_coordinates(distance,clipped[:,::-1].T,order=1,mode='nearest')+np.linalg.norm(uv-clipped,axis=1))*visibility;tips=project((control@M[:3,:3].T+M[:3,3])@C[:3,:3].T+C[:3,3]);errors=[silhouette/np.sqrt(len(v))/2.,(tips[0]-targets[0])/18.,(tips[1]-targets[1])/(28. if held else 18.),x[:3]/.025,x[3:]/.4]
   if previous is not None:errors.append(np.array([(Rotation.from_matrix(previous[:3,:3]).inv()*Rotation.from_matrix(M[:3,:3])).magnitude()/.6]))
   return np.concatenate([np.asarray(x).ravel() for x in errors])
  def overlap(M):
   pc=(vertices@M[:3,:3].T+M[:3,3])@C[:3,:3].T+C[:3,3];uv=np.rint(project(pc)*.5).astype(np.int32);ok=(pc[:,2]>.05)&(uv[:,0]>=0)&(uv[:,0]<512)&(uv[:,1]>=0)&(uv[:,1]<384);render=np.zeros((384,512),np.uint8);render[uv[ok,1],uv[ok,0]]=255;render=cv2.morphologyEx(render,cv2.MORPH_CLOSE,np.ones((3,3),np.uint8));contours,_=cv2.findContours(render,cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_SIMPLE);cv2.drawContours(render,contours,-1,255,-1);render=render>0;return float((render&small&valid).sum()/max(1,((render|small)&valid).sum()))
  fit=least_squares(residual,np.zeros(6),bounds=([-0.035]*3+[-.5]*3,[.035]*3+[.5]*3),loss='soft_l1',max_nfev=35);M=transform(fit.x);before,after=overlap(seed),overlap(M);accepted=after>=before-.005
  if not accepted:M=seed
  out['object_to_world'][i,b]=M;previous=M;details.append({'frame':i,'translation_change_m':float(np.linalg.norm(fit.x[:3])),'rotation_change_rad':float(np.linalg.norm(fit.x[3:])),'human_handle_observed':bool(held),'source_iou_before':before,'source_iou_after':after,'extra_fit_accepted':accepted})
  if i%25==0:print('Brush source evidence fitted',i,'/',T,flush=True)
 records.append({'object':'mini_sweeper','method':'Fresh silhouette distance field plus bristle-head and handle evidence; observed human thumb/index constrains the occluded handle; bounded35mm/0.5rad adjustment','frames':details})
# A released cup rests upright; use its last visible resting samples, not a drifting hand-relative pose.
if cfg['episode']==20:
 b=names.index('blue_cup');g=next(g for g in cfg['grasps'] if g['object']==b);start=g['end']+4;M=out['object_to_world'][start:,b].copy();position=np.median(M[:,:3,3],axis=0);rotation=Rotation.from_matrix(M[:,:3,:3]).mean().as_matrix();mesh=trimesh.load(rec/'mesh_priors/blue_cup.ply',process=False);vertices=np.asarray(mesh.vertices);z=(vertices@rotation.T)[:,2];position[2]=.6505-z.min()
 for i in range(start,T):out['object_to_world'][i,b,:3,3]=position;out['object_to_world'][i,b,:3,:3]=rotation
 records.append({'object':'blue_cup','source_supported_rest_frames':[start,T-1],'pose_estimated_from_visible_final_source_samples':True,'table_nonpenetration_applied':True})
assert all(np.isfinite(v).all() for v in out.values());np.savez_compressed(rec/'scene_motion_physical.npz',**out);save(marker,{'updated_utc':utc(),'source_frames':T,'old_reconstruction_preserved_separately':True,'records':records,'requires_fresh_source_projection_review':True,'evaluation_ground_truth_used':False})
