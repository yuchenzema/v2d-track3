"""Recover the initially missed left hand from source-image crops and calibrated depth."""
from pathlib import Path
import argparse,json,shutil,time
import cv2,numpy as np
from scipy.signal import savgol_filter
from scipy.spatial.transform import Rotation

parser=argparse.ArgumentParser();parser.add_argument('--apply',action='store_true');a=parser.parse_args();r=Path('/home/yu_eli/v2d-track3/runs/episode_000019_newton_batch_20261015');rec=r/'reconstruction';assert not (r/'coherent-input-checksums.json').exists();crop=np.load(r/'research/left-hand-cropped-detections.npz');flow=np.load(r/'research/left-hand-frame0-flow.npz');measure={k:v.copy() for k,v in np.load(rec/'hand_measurements.npz').items()};motion=np.load(rec/'scene_motion_physical.npz');camera=motion['camera_to_world'];H=motion['hand_joint_positions'].copy();intr=json.loads((rec/'camera_intrinsics.json').read_text());K=np.array([[intr['fx'],0,intr['cx']],[0,intr['fy'],intr['cy']],[0,0,1.]])
u=crop['uv'].copy();relative=crop['relative_xyz'].copy();u[0]=u[1]+flow['median_displacement'];u[0,flow['valid']]=flow['points'][flow['valid']];relative[0]=relative[1];fitted=np.full((31,21,3),np.nan);records=[];recovered_stereo=np.load(r/'research/early-hand-repair-stereo.npz');recovered_depths=recovered_stereo['depth'];recovered_valid=recovered_stereo['observed']
for i in range(31):
 if not np.isfinite(u[i]).all():continue
 ok,rv,tv=cv2.solvePnP(relative[i],u[i],K,None,flags=cv2.SOLVEPNP_SQPNP)
 if not ok:continue
 pc=relative[i]@Rotation.from_rotvec(rv[:,0]).as_matrix().T+tv[:,0];projected=cv2.projectPoints(relative[i],rv,tv,K,None)[0][:,0];error=float(np.median(np.linalg.norm(projected-u[i],axis=1)))
 if error>20:continue
 depth=recovered_depths[i];stereo=recovered_valid[i];image=cv2.imread(str(rec/f'frames/{i:06d}.jpg'));hsv=cv2.cvtColor(image,cv2.COLOR_BGR2HSV);skin=(hsv[:,:,0]<30)&(hsv[:,:,1]>35)&(hsv[:,:,1]<185)&(image[:,:,2]>image[:,:,1]*1.03)&(image[:,:,1]>image[:,:,0]*1.04);samples=[];indices=[];fallback=[];fallback_indices=[]
 for j,(xx,yy) in enumerate(u[i]):
  x,y=int(round(xx)),int(round(yy))
  if not (3<=x<1021 and 3<=y<765):continue
  patch=depth[y-3:y+4,x-3:x+4];valid=(patch>.2)&(patch<1.)&skin[y-3:y+4,x-3:x+4];observed=valid&stereo[y-3:y+4,x-3:x+4]
  if observed.sum()>=4:samples.append(np.median(patch[observed]));indices.append(j)
  if valid.sum()>=6:fallback.append(np.median(patch[valid]));fallback_indices.append(j)
 if len(samples)>=4:offset=float(np.median(np.array(samples)-pc[indices,2]))+.004;method='stereo on visible skin'
 elif len(fallback)>=5:offset=float(np.clip(np.median(np.array(fallback)-pc[fallback_indices,2])+.004,-.25*pc[0,2],.25*pc[0,2]));method='sparse metric skin depth, translation bounded by hand-scale PnP'
 elif error<12 and .25<pc[0,2]<.8:offset=0.;method='metric hand-scale PnP'
 else:continue
 z=pc[:,2]+offset
 if np.any(z<.18) or z[0]>.95:continue
 pc=np.c_[(u[i,:,0]-K[0,2])/K[0,0]*z,(u[i,:,1]-K[1,2])/K[1,1]*z,z];fitted[i]=pc@camera[i,:3,:3].T+camera[i,:3,3];records.append({'source_frame':i,'depth_method':method,'median_pnp_error_px':error,'wrist_depth_m':float(z[0])})
valid=np.isfinite(fitted[:,0,0]);assert valid[:20].sum()>=15
for j in range(21):
 for k in range(3):fitted[:,j,k]=np.interp(np.arange(31),np.flatnonzero(valid),fitted[valid,j,k])
filtered=savgol_filter(fitted,7,2,axis=0);H[:24,0]=filtered[:24]
for i in range(24,28):
 blend=(28-i)/5.;H[i,0]=blend*filtered[i]+(1-blend)*H[i,0]
measure['uv'][:20,0]=u[:20];measure['relative_xyz'][:20,0]=relative[:20];measure['observed'][:20,0]=True;measure['score'][:20,0]=crop['score'][:20];measure['score'][0,0]=crop['score'][1]*.5
out=r/'research/early-left-hand-repair-candidate.npz';np.savez_compressed(out,hand_joint_positions=H,**{f'measurement_{k}':v for k,v in measure.items()});images=[];cap=cv2.VideoCapture(str(rec/'calibrated.mp4'))
for i in range(20):
 ok,image=cap.read();assert ok;C=np.linalg.inv(camera[i]);p=H[i,0];pc=p@C[:3,:3].T+C[:3,3];uv=np.c_[intr['fx']*pc[:,0]/pc[:,2]+intr['cx'],intr['fy']*pc[:,1]/pc[:,2]+intr['cy']];old=motion['hand_joint_positions'][i,0,0];old=C[:3,:3]@old+C[:3,3];old_uv=np.array([intr['fx']*old[0]/old[2]+intr['cx'],intr['fy']*old[1]/old[2]+intr['cy']]);cv2.circle(image,tuple(old_uv.round().astype(int)),9,(255,0,255),2)
 for finger in range(5):
  chain=[0,*range(1+4*finger,5+4*finger)]
  for x,y in zip(chain[:-1],chain[1:]):cv2.line(image,tuple(uv[x].round().astype(int)),tuple(uv[y].round().astype(int)),(0,255,0),2)
 cv2.putText(image,f'Frame {i}: recovered left (green), old wrist (magenta)',(10,35),0,.65,(255,255,255),2);images.append(cv2.resize(image,(320,240)))
cap.release();cv2.imwrite(str(r/'research/early-left-hand-repair-review.jpg'),np.concatenate([np.concatenate(images[i:i+4],axis=1) for i in range(0,20,4)]));report={'method':__doc__,'replaced_measurement_frames':[0,19],'replaced_3d_hand_frames':[0,27],'source_frame_15_false_duplicate_rejected':True,'frame_0_method':'Backward pyramidal Lucas-Kanade from source frame 1; 14/21 points pass <3 px forward-backward error; remaining points use the robust observed palm displacement and frame-1 relative pose.','cropped_landmarker_detection_frames':list(range(1,20)),'world_measurements':records,'other_hand_and_all_object_and_camera_poses_preserved':True,'applied':a.apply}
if a.apply:
 backup=r/f'research/early-left-hand-repair-{time.time_ns()}';backup.mkdir();shutil.copy2(rec/'hand_measurements.npz',backup/'hand_measurements.npz');np.savez_compressed(rec/'hand_measurements.npz',**measure)
 for name in ['scene_motion_physical.npz','scene_motion.npz','scene_motion_raw.npz']:
  path=rec/name
  if not path.exists():continue
  shutil.copy2(path,backup/name);old={k:v.copy() for k,v in np.load(path).items()};updated=old['hand_joint_positions'].copy()
  for i in range(28):
   camera_points=(H[i,0]-camera[i,:3,3])@camera[i,:3,:3];updated[i,0]=camera_points@old['camera_to_world'][i,:3,:3].T+old['camera_to_world'][i,:3,3]
  old['hand_joint_positions']=updated;old['observed']=measure['observed'];np.savez_compressed(path,**old)
 report['backup']=str(backup.relative_to(r));meta=r/'reconstruction/scene-reconstruction.json';value=json.loads(meta.read_text());value['hand_observed_frames']=measure['observed'].sum(0).tolist();value['early_left_hand_repair']='research/early-left-hand-repair.json';meta.write_text(json.dumps(value,indent=2))
(r/'research/early-left-hand-repair.json').write_text(json.dumps(report,indent=2));print(json.dumps(report,indent=2),flush=True)
