"""Continue each released wrist using observed human motion in the fitted grasp frame.

Held robot poses, object trajectories and timestamps stay fixed. Free wrist paths
are continuous across each handover, with bounded increments on the source grid.
"""
import json,os
from pathlib import Path
import numpy as np,pyarrow as pa,pyarrow.parquet as pq,trimesh
from scipy.spatial import cKDTree
from scipy.spatial.transform import Rotation,Slerp
from scipy.ndimage import median_filter,gaussian_filter1d
from scipy.optimize import least_squares
from robotic_grounding.retarget.retarget_utils import setup_sharpa_kinematics

R=Path('/episode');cfg=json.loads((R/'episode-config.json').read_text());reference=os.environ.get('V2D_CONTINUOUS_REFERENCE',cfg.get('training_reference','track3_coherent_variant'))
if (R/'coherent-input-checksums.json').exists():
 assert os.environ.get('V2D_ALLOW_FROZEN_ALTERNATE')=='1' and reference!=cfg.get('training_reference','track3_coherent_variant')
path=next((R/f'human_motion_data/{reference}/processed').rglob('motion.parquet'));row=pq.ParquetFile(path).read().to_pylist()[0]
raw=pq.ParquetFile(next((R/'human_motion_data/track3_video/processed').rglob('motion.parquet'))).read().to_pylist()[0]
T=cfg['frames'];H=np.load(R/'reconstruction/scene_motion_physical.npz')['hand_joint_positions'];P=np.array(row['object_body_position']);Q=np.array(row['object_body_wxyz']);B=len(cfg['objects']);object_rotations=Rotation.from_quat(Q.reshape(-1,4),scalar_first=True).as_matrix().reshape(T,B,3,3);meshes=[trimesh.load(R/f'reconstruction/mesh_priors/{name}.ply',process=False) for name in cfg['objects']];trees=[cKDTree(mesh.vertices) for mesh in meshes];reports=[]

def limit_positions(positions,endpoint_fixed,maximum):
 values=positions.copy()
 for iteration in range(200):
  for i in range(1,len(values)-(1 if endpoint_fixed else 0)):
   delta=values[i]-values[i-1];distance=np.linalg.norm(delta)
   if distance>maximum:values[i]=values[i-1]+delta*(maximum/distance)
  if endpoint_fixed:
   for i in range(len(values)-2,0,-1):
    delta=values[i]-values[i+1];distance=np.linalg.norm(delta)
    if distance>maximum:values[i]=values[i+1]+delta*(maximum/distance)
  if np.linalg.norm(np.diff(values,axis=0),axis=1).max(initial=0)<=maximum+1e-8:break
 return values

def limit_rotations(rotations,endpoint_fixed,maximum=.35):
 values=list(rotations)
 for iteration in range(200):
  for i in range(1,len(values)-(1 if endpoint_fixed else 0)):
   delta=(values[i-1].inv()*values[i]).as_rotvec();angle=np.linalg.norm(delta)
   if angle>maximum:values[i]=values[i-1]*Rotation.from_rotvec(delta*(maximum/angle))
  if endpoint_fixed:
   for i in range(len(values)-2,0,-1):
    delta=(values[i+1].inv()*values[i]).as_rotvec();angle=np.linalg.norm(delta)
    if angle>maximum:values[i]=values[i+1]*Rotation.from_rotvec(delta*(maximum/angle))
  if max(((values[i-1].inv()*values[i]).magnitude() for i in range(1,len(values))),default=0)<=maximum+1e-8:break
 return Rotation.from_quat([x.as_quat() for x in values])

for si,side in enumerate(['left','right']):
 w=np.array(row[f'robot_{side}_wrist_position']);wq=np.array(row[f'robot_{side}_wrist_wxyz']);f=np.array(row[f'robot_{side}_finger_joints']);free_fingers=np.array(raw[f'robot_{side}_finger_joints']);frames=np.array(row[f'robot_{side}_frames']);names=row[f'{side}_robot_frame_names'];human=H[:,si];y=human[:,9]-human[:,0];y/=np.maximum(1e-12,np.linalg.norm(y,axis=1)[:,None]);x=human[:,5]-human[:,17];x-=y*np.sum(x*y,axis=1)[:,None];x/=np.maximum(1e-12,np.linalg.norm(x,axis=1)[:,None]);palm=np.stack([x,y,np.cross(x,y)],axis=-1);grasps=sorted([g for g in cfg['grasps'] if g['side']==side],key=lambda g:g['start'])
 for gi,g in enumerate(grasps):
  start=g['end'];next_start=grasps[gi+1]['start'] if gi+1<len(grasps) else None;stop=next_start if next_start is not None else T-1
  if stop-start<2:continue
  old=w[start:stop+1].copy();oldq=wq[start:stop+1].copy();base=Rotation.from_quat(wq[start],scalar_first=True);target=w[start]+human[start:stop+1,0]-human[start,0];rotations=Rotation.from_matrix(palm[start:stop+1]@palm[start].T@base.as_matrix());rotations=Rotation.from_matrix([rotations[max(0,i-1):min(len(rotations),i+2)].mean().as_matrix() for i in range(len(rotations))]);rotations=Rotation.from_quat(np.r_[base.as_quat()[None],rotations[1:].as_quat()])
  if next_start is not None:
   t=np.linspace(0,1,len(target));blend=t*t*(3-2*t);target+=blend[:,None]*(w[stop]-target[-1]);correction=Rotation.from_quat(wq[stop],scalar_first=True)*rotations[-1].inv();rotations=Rotation.from_rotvec(blend[:,None]*correction.as_rotvec())*rotations
  # The fitted wrist offset is local to contact. Restore the independently
  # observed free-hand retarget gradually once the hand leaves the object;
  # carrying that offset through the whole free interval biases image alignment.
  t=np.clip(np.arange(len(target))/10.,0.,1.);blend=t*t*(3-2*t);target=(1-blend[:,None])*target+blend[:,None]*old;original_rotations=Rotation.from_quat(oldq,scalar_first=True);delta=original_rotations*rotations.inv();rotations=Rotation.from_rotvec(blend[:,None]*delta.as_rotvec())*rotations
  if os.environ.get('V2D_RELEASE_PATH_MODEL')=='pinhole':
   # Smooth uncertain depth along the observed source-camera rays. Cartesian
   # smoothing can move a correctly detected hand sideways in the video.
   cameras=np.load(R/'reconstruction/scene_motion_physical.npz')['camera_to_world'][start:stop+1];pc=np.einsum('tji,tj->ti',cameras[:,:3,:3],old-cameras[:,:3,3]);bearing=pc[:,:2]/pc[:,2:];bearing=limit_positions(bearing,next_start is not None,.06);rays=np.c_[bearing,np.ones(len(bearing))];depth=gaussian_filter1d(median_filter(pc[:,2],size=min(5,len(pc)//2*2+1)),1.);fade=np.clip(1-np.arange(len(pc))/8.,0,1);depth+=fade*(pc[0,2]-depth[0]);depth[0]=pc[0,2]
   if next_start is not None:
    fade=np.clip(1-np.arange(len(pc))[::-1]/8.,0,1);depth+=fade*(pc[-1,2]-depth[-1]);depth[-1]=pc[-1,2]
   indices=np.arange(1,len(depth)-(1 if next_start is not None else 0));prior=depth.copy()
   def world_positions(z):return np.einsum('tij,tj->ti',cameras[:,:3,:3],rays*z[:,None])+cameras[:,:3,3]
   def depth_loss(x):
    z=prior.copy();z[indices]=x;positions=world_positions(z)
    return np.r_[(z-prior)/.015,np.maximum(0,np.linalg.norm(np.diff(positions,axis=0),axis=1)-.04)/.001,np.diff(z,2)/.01]
   fit=least_squares(depth_loss,np.clip(depth[indices],.180001,.999999),bounds=(.18,1.),max_nfev=200,x_scale=.02);depth[indices]=fit.x;target=world_positions(depth);rotations=original_rotations
  bound=max(.04,float(np.linalg.norm(target[-1]-target[0])/(len(target)-1))*1.05) if next_start is not None else .04
  if os.environ.get('V2D_RELEASE_PATH_MODEL')!='pinhole':target=limit_positions(target,next_start is not None,bound)
  angular_bound=max(.35,float((base.inv()*Rotation.from_quat(wq[stop],scalar_first=True)).magnitude()/(len(target)-1))*1.05) if next_start is not None else .35;rotations=limit_rotations(rotations,next_start is not None,angular_bound)
  w[start+1:stop+(0 if next_start is not None else 1)]=target[1:len(target)-(1 if next_start is not None else 0)];wq[start+1:stop+(0 if next_start is not None else 1)]=rotations[1:len(target)-(1 if next_start is not None else 0)].as_quat(scalar_first=True)
  open_frame=min(stop-1 if next_start is not None else stop,start+5);open_posture=free_fingers[open_frame]
  for i in range(start+1,min(stop if next_start is not None else stop+1,start+4)):
   blend=min(1.,(i-start)/3);f[i]=(1-blend)*f[start]+blend*open_posture
  # Keep the source retarget's subsequent free-hand articulation, rather than
  # extending a grasp posture into the observed release.
  if stop-start>4:f[start+4:stop+(0 if next_start is not None else 1)]=free_fingers[start+4:stop+(0 if next_start is not None else 1)]
  reports.append({'side':side,'release_frame':start,'next_grasp_frame':next_start,'free_interval':[start+1,stop-1 if next_start is not None else stop],'maximum_wrist_step_before_m':float(np.linalg.norm(np.diff(old,axis=0),axis=1).max()),'maximum_wrist_step_after_m':float(np.linalg.norm(np.diff(w[start:stop+1],axis=0),axis=1).max()),'maximum_rotation_step_before_rad':float((Rotation.from_quat(oldq[:-1],scalar_first=True).inv()*Rotation.from_quat(oldq[1:],scalar_first=True)).magnitude().max()),'maximum_rotation_step_after_rad':float((Rotation.from_quat(wq[start:stop],scalar_first=True).inv()*Rotation.from_quat(wq[start+1:stop+1],scalar_first=True)).magnitude().max())})
 # Every held wrist and held finger target remains exactly unchanged.
 for g in grasps:
  interval=slice(g['start'],g['end']+1)
  assert np.array_equal(w[interval],np.array(row[f'robot_{side}_wrist_position'])[interval])
  assert np.array_equal(wq[interval],np.array(row[f'robot_{side}_wrist_wxyz'])[interval])
  assert np.array_equal(f[interval],np.array(row[f'robot_{side}_finger_joints'])[interval])
 kin=setup_sharpa_kinematics(side,frequency=50.,max_iter=100,frame_tasks_converged_threshold=2e-6)
 for i in range(T):
  kin.configuration.update(np.r_[w[i],wq[i,[1,2,3,0]],f[i]])
  for j,name in enumerate(names):
   tr=kin.configuration.get_transform_frame_to_world(name);frames[i,j]=np.r_[tr.translation,Rotation.from_matrix(tr.rotation).as_quat(scalar_first=True)]
 cp=np.zeros((T,5,3));cn=np.zeros_like(cp);pid=np.zeros((T,5),int);tips=[names.index(f'{side}_{finger}_tip_site') for finger in ['thumb','index','middle','ring','pinky']]
 for i in range(T):
  active=[g['object'] for g in grasps if max(0,g['start']-3)<=i<=g['end']]
  for j,tip in enumerate(tips):
   found=[]
   for b in active:
    distance,index=trees[b].query((frames[i,tip,:3]-P[i,b])@object_rotations[i,b]);found.append((distance,b,index))
   if not found:continue
   distance,b,index=min(found)
   if distance>.025:continue
   cp[i,j]=object_rotations[i,b]@meshes[b].vertices[index]+P[i,b];normal=object_rotations[i,b]@meshes[b].vertex_normals[index];cn[i,j]=normal/(np.linalg.norm(normal)+1e-12);pid[i,j]=b+1
 for key,value in [('wrist_position',w),('wrist_wxyz',wq),('finger_joints',f),('frames',frames)]:row[f'robot_{side}_{key}']=value.tolist()
 for key,value in [('positions',cp),('normals',cn),('part_ids',pid)]:row[f'mano_{side}_object_contact_{key}']=value.tolist()
assert np.array_equal(P,np.array(row['object_body_position'])) and np.array_equal(Q,np.array(row['object_body_wxyz']))
pq.write_table(pa.Table.from_pylist([row]),path,compression='zstd');(R/f'research/{reference}-continuous-release.json').write_text(json.dumps({'method':__doc__,'reference':reference,'objects_unchanged':True,'held_wrist_and_finger_targets_unchanged':True,'source_timestamps_unchanged':True,'hands':reports},indent=2));owner=R.stat();os.chown(path,owner.st_uid,owner.st_gid);print(reports,flush=True)
