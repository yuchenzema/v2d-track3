"""Calibrate post-release robot wrist motion to the last fitted video grasp.

Object trajectories are unchanged. Observed human wrist displacements and palm
rotations continue from the fitted grasp, avoiding a blend between incompatible
contact-constrained and free-hand depth estimates.
"""
import json,shutil
from pathlib import Path
import numpy as np,pyarrow as pa,pyarrow.parquet as pq,trimesh
from scipy.spatial import cKDTree
from scipy.spatial.transform import Rotation
from robotic_grounding.retarget.retarget_utils import setup_sharpa_kinematics
R=Path('/episode');cfg=json.loads((R/'episode-config.json').read_text());assert cfg['episode']==8
source=next((R/'human_motion_data/track3_collision_variant/processed').rglob('motion.parquet'));row=pq.ParquetFile(source).read().to_pylist()[0];motion=np.load(R/'reconstruction/scene_motion_physical.npz');H=motion['hand_joint_positions'];T=cfg['frames'];P=np.array(row['object_body_position']);Q=np.array(row['object_body_wxyz']);objects=Rotation.from_quat(Q.reshape(-1,4),scalar_first=True).as_matrix().reshape(T,len(cfg['objects']),3,3);meshes=[trimesh.load(R/f'reconstruction/mesh_priors/{name}.ply',process=False) for name in cfg['objects']];trees=[cKDTree(m.vertices) for m in meshes];reports=[]
for side_index,side in enumerate(['left','right']):
 w=np.array(row[f'robot_{side}_wrist_position']);q=np.array(row[f'robot_{side}_wrist_wxyz']);f=np.array(row[f'robot_{side}_finger_joints']);frames=np.array(row[f'robot_{side}_frames']);names=row[f'{side}_robot_frame_names']
 if side=='right':
  end=201;before=w.copy();human=H[:,side_index];y=human[:,9]-human[:,0];y/=np.linalg.norm(y,axis=1)[:,None];x=human[:,5]-human[:,17];x-=y*np.sum(x*y,axis=1)[:,None];x/=np.linalg.norm(x,axis=1)[:,None];palm=np.stack([x,y,np.cross(x,y)],axis=-1);base=Rotation.from_quat(q[end],scalar_first=True).as_matrix();base_position=w[end].copy()
  w[end:]=base_position+human[end:,0]-human[end,0]
  rotations=Rotation.from_matrix(palm[end:]@palm[end].T@base);filtered=Rotation.from_matrix([rotations[max(0,j-1):min(len(rotations),j+2)].mean().as_matrix() for j in range(len(rotations))]);q[end+1:]=filtered[1:].as_quat(scalar_first=True)
  open_posture=f[208].copy();held=f[199].copy()
  for i in range(199,T):blend=min(1.,max(0.,(i-199)/2));f[i]=(1-blend)*held+blend*open_posture
  reports.append({'side':side,'release_source_frame':end,'maximum_post_release_wrist_step_before_m':float(np.linalg.norm(np.diff(before[end:],axis=0),axis=1).max()),'maximum_post_release_wrist_step_after_m':float(np.linalg.norm(np.diff(w[end:],axis=0),axis=1).max()),'open_posture_source_frame':208,'finger_opening_interval':[199,201]})
 kin=setup_sharpa_kinematics(side,frequency=50.,max_iter=100,frame_tasks_converged_threshold=2e-6)
 for i in range(T):
  kin.configuration.update(np.r_[w[i],q[i,[1,2,3,0]],f[i]])
  for j,name in enumerate(names):tr=kin.configuration.get_transform_frame_to_world(name);frames[i,j]=np.r_[tr.translation,Rotation.from_matrix(tr.rotation).as_quat(scalar_first=True)]
 cp=np.zeros((T,5,3));cn=np.zeros_like(cp);pid=np.zeros((T,5),int);tips=[names.index(f'{side}_{finger}_tip_site') for finger in ['thumb','index','middle','ring','pinky']]
 for i in range(T):
  active=[g['object'] for g in cfg['grasps'] if g['side']==side and max(0,g['start']-3)<=i<=g['end']]
  for j,tip in enumerate(tips):
   found=[]
   for b in active:
    distance,index=trees[b].query((frames[i,tip,:3]-P[i,b])@objects[i,b]);found.append((distance,b,index))
   if not found:continue
   distance,b,index=min(found)
   if distance>.025:continue
   cp[i,j]=objects[i,b]@meshes[b].vertices[index]+P[i,b];normal=objects[i,b]@meshes[b].vertex_normals[index];cn[i,j]=normal/(np.linalg.norm(normal)+1e-12);pid[i,j]=b+1
 for key,value in [('wrist_position',w),('wrist_wxyz',q),('finger_joints',f),('frames',frames)]:row[f'robot_{side}_{key}']=value.tolist()
 for key,value in [('positions',cp),('normals',cn),('part_ids',pid)]:row[f'mano_{side}_object_contact_{key}']=value.tolist()
destination=R/'human_motion_data/track3_release_calibrated';out=destination/'processed'/f"sequence_id={cfg['sequence']}"/'robot_name=sharpa_wave';out.mkdir(parents=True,exist_ok=True);pq.write_table(pa.Table.from_pylist([row]),out/'motion.parquet',compression='zstd');stage=destination/'reconstructed_stage';stage.mkdir(exist_ok=True)
for path in (R/'human_motion_data/track3_collision_variant/reconstructed_stage').glob('*.usda'):shutil.copy2(path,stage/path.name)
old=pq.ParquetFile(source).read().to_pylist()[0];assert np.array_equal(np.array(old['object_body_position']),np.array(row['object_body_position'])) and np.array_equal(np.array(old['object_body_wxyz']),np.array(row['object_body_wxyz']))
(R/'research/release-reference-calibration.json').write_text(json.dumps({'method':__doc__,'source_reference':'track3_collision_variant','new_reference':'track3_release_calibrated','object_poses_unchanged':True,'all_source_timestamps_preserved':True,'contact_targets_end_at_observed_release':True,'hands':reports},indent=2));print(reports,flush=True)
