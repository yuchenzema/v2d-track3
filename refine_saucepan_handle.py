"""Resolve the occluded saucepan handle using source hand contact and visible pixels."""
from pathlib import Path
import importlib.util,json
import cv2,numpy as np,pyarrow.parquet as pq,trimesh
from scipy.ndimage import distance_transform_edt,map_coordinates
from scipy.spatial.transform import Rotation
r=Path('/home/yu_eli/v2d-track3/runs/episode_000028_newton_batch_20261015');cfg=json.loads((r/'episode-config.json').read_text());assert not (r/'coherent-input-checksums.json').exists();rec=r/'reconstruction';motion={k:v.copy() for k,v in np.load(rec/'scene_motion_physical.npz').items()};measure=np.load(rec/'hand_measurements.npz');row=pq.ParquetFile(next((r/'human_motion_data/track3_collision_variant').rglob('motion.parquet'))).read().to_pylist()[0];P=np.array(row['object_body_position'])[:,1];R=Rotation.from_quat(np.array(row['object_body_wxyz'])[:,1],scalar_first=True).as_matrix();mesh=trimesh.load(rec/'mesh_priors/white_pot.ply',process=False);vertices=np.asarray(mesh.vertices);sample=vertices[np.linspace(0,len(vertices)-1,min(1800,len(vertices))).astype(int)];handle=vertices[vertices[:,2]>.14];handle=handle[np.linspace(0,len(handle)-1,min(180,len(handle))).astype(int)];spec=importlib.util.spec_from_file_location('handle_source_geometry',r/'scripts/source_geometry.py');g=importlib.util.module_from_spec(spec);spec.loader.exec_module(g);angles=np.linspace(0,2*np.pi,72,endpoint=False);A=Rotation.from_rotvec(np.c_[angles,np.zeros((len(angles),2))]).as_matrix();T=cfg['frames'];costs=[];rotations=[];notes=[]
for i in range(T):
 rotations.append(R[i][None]@A);world=sample[None]@rotations[-1].transpose(0,2,1)+P[i];C=np.linalg.inv(motion['camera_to_world'][i]);pc=world@C[:3,:3].T+C[:3,3];uv=(pc[...,:2]/pc[...,2:]*[g.K[0,0],g.K[1,1]]+[g.K[0,2],g.K[1,2]])*.5;mask=cv2.resize(cv2.imread(str(rec/f'masks/2/{i:06d}.png'),0),(512,384),interpolation=cv2.INTER_NEAREST)>0;valid=~g.human_occlusion(measure,i);valid&=cv2.resize(cv2.imread(str(rec/f'masks/1/{i:06d}.png'),0),(512,384),interpolation=cv2.INTER_NEAREST)==0;dist=distance_transform_edt(~mask);clipped=np.stack([np.clip(uv[...,0],0,511),np.clip(uv[...,1],0,383)],axis=-1);outside=map_coordinates(dist,clipped[...,::-1].reshape(-1,2).T,order=1,mode='nearest').reshape(len(A),-1);visible=map_coordinates(valid.astype(float),clipped[...,::-1].reshape(-1,2).T,order=0,mode='nearest').reshape(len(A),-1);outside+=np.linalg.norm(uv-clipped,axis=-1);silhouette_cost=(np.minimum(outside,25.)**2*visible).sum(1)/np.maximum(1.,visible.sum(1))/9.;contact_cost=np.zeros(len(A));sides=[]
 for si,side in enumerate(['left','right']):
  if not any(x['side']==side and x['object']==1 and x['start']<=i<=x['end'] for x in cfg['grasps']):continue
  if measure['observed'][i,si]:target=measure['uv'][i,si,[4,8]]*.5
  else:
   points=motion['hand_joint_positions'][i,si,[4,8]];target=g.project(points@C[:3,:3].T+C[:3,3])*.5
  wh=handle[None]@rotations[-1].transpose(0,2,1)+P[i];ch=wh@C[:3,:3].T+C[:3,3];uh=(ch[...,:2]/ch[...,2:]*[g.K[0,0],g.K[1,1]]+[g.K[0,2],g.K[1,2]])*.5;distance=np.linalg.norm(uh[:,None]-target[None,:,None],axis=-1).min(2);contact_cost+=np.mean(np.minimum(distance,60.)**2,axis=1)/16.;sides.append(side)
 cost=silhouette_cost+contact_cost
 if i<=70:cost[1:]=np.inf
 costs.append(cost);notes.append({'frame':i,'contact_sides':sides,'local_best_yaw_rad':float(angles[int(np.argmin(cost))])})
 if i%25==0:print(i,'handle pose cost prepared',flush=True)
costs=np.array(costs);rotations=np.array(rotations);dp=costs[0].copy();parents=[]
for i in range(1,T):
 cosine=np.clip((rotations[i-1].reshape(len(A),9)@rotations[i].reshape(len(A),9).T-1)*.5,-1,1);angular=np.arccos(cosine);normal_change=float(np.arccos(np.clip(R[i-1,:,0]@R[i,:,0],-1,1)));maximum=max(.3,normal_change+.08)
 if i<=70:maximum=max(maximum,float((Rotation.from_matrix(R[i-1]).inv()*Rotation.from_matrix(R[i])).magnitude())+.001)
 transition=(angular/.15)**2;transition[angular>maximum]=np.inf;scores=dp[:,None]+transition;parent=np.argmin(scores,axis=0);dp=costs[i]+scores[parent,np.arange(len(A))];assert np.isfinite(dp).any(),i;parents.append(parent)
path=[int(np.argmin(dp))]
for parent in parents[::-1]:path.append(int(parent[path[-1]]))
path=path[::-1];chosen=rotations[np.arange(T),path];candidate=motion['object_to_world'].copy();candidate[:,1,:3,3]=P;candidate[:,1,:3,:3]=chosen;motion['object_to_world']=candidate;np.savez_compressed(rec/'scene_motion_handle_candidate.npz',**motion);delta=(Rotation.from_matrix(R).inv()*Rotation.from_matrix(chosen)).magnitude();step=(Rotation.from_matrix(chosen[:-1]).inv()*Rotation.from_matrix(chosen[1:])).magnitude();report={'method':__doc__,'object':'white_pot','axis_local':[1,0,0],'handle_local_positive_z':True,'preserved_early_pose_frames':[0,70],'uses_confidential_evaluation_poses':False,'normal_axis_unchanged':True,'translations_and_scale_unchanged_from_physical_reference':True,'maximum_yaw_correction_rad':float(delta.max()),'maximum_orientation_step_rad':float(step.max()),'frame_yaw_rad':angles[path].tolist(),'source_cost_and_contact_observations':notes};(r/'research/saucepan-handle-candidate.json').write_text(json.dumps(report,indent=2));print({k:v for k,v in report.items() if k not in ['frame_yaw_rad','source_cost_and_contact_observations']},flush=True)
