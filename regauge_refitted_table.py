"""Apply a robust global scene gauge from independently refitted tabletop circles."""
from pathlib import Path
import json,numpy as np,shutil,time,fcntl,cv2
from scipy.spatial.transform import Rotation
R=Path('/home/yu_eli/v2d-track3/runs/episode_000014_newton_batch_20261015');rec=R/'reconstruction';cfg=json.loads((R/'episode-config.json').read_text())
with (R/'research/collision-reference.lock').open('a') as lock:
 fcntl.flock(lock,fcntl.LOCK_EX);assert not (R/'coherent-input-checksums.json').exists();backup=R/f'research/table-circle-refit-{time.time_ns()}';backup.mkdir();old=np.load(rec/'table-camera-calibration.npz');data={k:old[k].copy() for k in old.files};new=np.load(rec/'refitted-table-circle-measurements.npz');S=[]
 for i in range(cfg['frames']):
  p=json.loads((rec/f'slam/poses/{i:06d}.json').read_text());M=np.eye(4);M[:3,:3]=Rotation.from_quat(p['rotation'],scalar_first=True).as_matrix();M[:3,3]=p['translation'];S.append(M)
 S=np.array(S);anchors=np.einsum('nij,nj->ni',S[:,:3,:3],new['table_centers_camera'])+S[:,:3,3];radius=float(np.median(new['measured_radii']));good=(abs(new['measured_radii']-radius)<.015)&(new['circle_inlier_fraction']>.2)
 for iteration in range(4):
  center=np.median(anchors[good],axis=0);dist=np.linalg.norm(anchors-center,axis=1);values=dist[good];cut=max(.025,float(np.median(values)+3.5*1.4826*np.median(abs(values-np.median(values)))));good&=dist<cut
 assert good.sum()>=12,good.sum();normals=np.einsum('nij,nj->ni',S[:,:3,:3],data['table_planes_camera'][:,:3]);up=np.median(normals[good],axis=0);up/=np.linalg.norm(up);good&=normals@up>np.cos(.15);assert good.sum()>=12;center=np.median(anchors[good],axis=0);up=np.median(normals[good],axis=0);up/=np.linalg.norm(up);x=S[0,:3,0].copy();x-=up*(x@up);x/=np.linalg.norm(x);world=np.eye(4);world[:3,:3]=np.stack([x,np.cross(up,x),up]);world[:3,3]=np.array([0.,0.,.65])-world[:3,:3]@center;camera=world@S;G=camera[0]@np.linalg.inv(data['camera_to_world'][0]);assert np.max(abs(camera-G@data['camera_to_world']))<1e-9
 residual=np.linalg.norm(anchors-center,axis=1);newradius=float(np.median(new['measured_radii'][good]));checks={}
 for p in rec.glob('scene_motion*.npz'):
  shutil.copy2(p,backup/p.name);a=np.load(p);out={k:a[k].copy() for k in a.files};before=np.linalg.inv(out['camera_to_world'])[:,None]@out['object_to_world'];out['camera_to_world']=G@out['camera_to_world'];out['object_to_world']=G@out['object_to_world'];out['hand_joint_positions']=out['hand_joint_positions']@G[:3,:3].T+G[:3,3];after=np.linalg.inv(out['camera_to_world'])[:,None]@out['object_to_world'];error=float(np.max(abs(before-after)));assert error<1e-9;checks[p.name]=error;np.savez_compressed(p,**out)
 for name in ['table-camera-calibration.npz','table-camera-calibration.json','camera-stability.json','rebuilt_scene.glb','tabletop_video_texture.png']:
  p=rec/name
  if p.exists():shutil.move(p,backup/name)
 for directory in ['track3_video','track3_coherent_variant','track3_collision_variant']:
  p=R/'human_motion_data'/directory
  if p.exists():p.rename(backup/directory)
 for p in (R/'research').glob('collision-*.json'):shutil.move(p,backup/p.name)
 for p in (R/'evaluation').glob('reference-source-fidelity*'):shutil.move(p,backup/p.name)
 data.update(camera_to_world=camera,table_centers_camera=new['table_centers_camera'],measured_radii=new['measured_radii'],circle_inlier_fraction=new['circle_inlier_fraction']);np.savez_compressed(rec/'table-camera-calibration.npz',**data)
 report=json.loads((backup/'table-camera-calibration.json').read_text());report.update(method=__doc__,support_radius_m=newradius,translation_source='Metric SLAM relative motion and robust static tabletop anchors; rejected inconsistent circle centers');(rec/'table-camera-calibration.json').write_text(json.dumps(report,indent=2));stability=json.loads((backup/'camera-stability.json').read_text());stability.update(method=__doc__,table_radius_m=newradius,retained_table_anchor_frames=np.flatnonzero(good).tolist(),median_table_anchor_residual_m=float(np.median(residual[good])),p95_table_anchor_residual_m=float(np.quantile(residual[good],.95)),circle_refit='reconstruction/refitted-table-circle-report.json',spatial_outlier_rejection=True);(rec/'camera-stability.json').write_text(json.dumps(stability,indent=2))
 cfg['reconstruction_visual_reviewed']=False;(R/'episode-config.json').write_text(json.dumps(cfg,indent=2));out={'method':__doc__,'source_projection_max_matrix_change':checks,'global_transform':G.tolist(),'new_table_radius_m':newradius,'retained_anchor_count':int(good.sum()),'median_residual_m':float(np.median(residual[good])),'p95_residual_m':float(np.quantile(residual[good],.95)),'backup':str(backup.relative_to(R))};(R/'evaluation/table-gauge-refit.json').write_text(json.dumps(out,indent=2));print(json.dumps(out,indent=2))
 # Show one fixed world table through the full recovered moving camera.
 intr=json.loads((rec/'camera_intrinsics.json').read_text());cap=cv2.VideoCapture(str(rec/'calibrated.mp4'));panels=[]
 for i in np.linspace(0,cfg['frames']-1,8).round().astype(int):
  cap.set(cv2.CAP_PROP_POS_FRAMES,int(i));ok,im=cap.read();assert ok;angles=np.linspace(0,2*np.pi,200);points=np.c_[newradius*np.cos(angles),newradius*np.sin(angles),np.full(len(angles),.65)];v=(points-camera[i,:3,3])@camera[i,:3,:3];uv=np.rint(v[:,:2]/v[:,2:]*[intr['fx'],intr['fy']]+[intr['cx'],intr['cy']]).astype(int);cv2.polylines(im,[uv],True,(0,255,0),3);panels.append(cv2.resize(im,(512,384)))
 cap.release();cv2.imwrite(str(rec/'robust-global-table-review.jpg'),np.concatenate([np.concatenate(panels[j:j+4],axis=1) for j in [0,4]]))
