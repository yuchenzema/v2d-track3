"""Calibrate the released nested stack from an unassisted gravity-settled pose.

The human releases the inner cup around102, then sets the stack down by126.
Only the training reference changes; learned rollout objects remain free bodies.
"""
from pathlib import Path
import shutil,json
import numpy as np,pyarrow as pa,pyarrow.parquet as pq
from scipy.spatial.transform import Rotation,Slerp
r=Path('/home/yu_eli/v2d-track3/runs/episode_000010_newton_grasp_curriculum_20261020');ref='track3_hollow_supported_candidate'
assert not (r/'coherent-input-checksums.json').exists();src=r/'human_motion_data/track3_hollow_grasp_candidate';dst=r/'human_motion_data'/ref;shutil.copytree(src,dst,dirs_exist_ok=True)
p=next(dst.rglob('motion.parquet'));row=pq.ParquetFile(p).read().to_pylist()[0];P0=np.array(row['object_body_position']);Q0=np.array(row['object_body_wxyz']);T,B=P0.shape[:2]
old=np.tile(np.eye(4),(T,B,1,1));old[:,:,:3,3]=P0;old[:,:,:3,:3]=Rotation.from_quat(Q0.reshape(-1,4),scalar_first=True).as_matrix().reshape(T,B,3,3);new=old.copy()
d=np.load(r/'evaluation/hollow-nesting-terminal-rest-capacity2048-states.npz');names=list(d['body_names']);settled=[]
for n in ['blue_cup','beige_cup']:
 q=d['body_poses'][-1,names.index(f'track3_{n}/object')];M=np.eye(4);M[:3,3]=q[:3];M[:3,:3]=Rotation.from_quat(q[3:]).as_matrix();settled.append(M)
relative=np.linalg.inv(settled[0])@settled[1]
def blend(A,B,weight):
 b=float(np.clip(weight,0,1));b=b*b*(3-2*b);M=np.eye(4);M[:3,3]=(1-b)*A[:3,3]+b*B[:3,3];M[:3,:3]=Slerp([0,1],Rotation.from_matrix([A[:3,:3],B[:3,:3]]))([b]).as_matrix()[0];return M
for i in range(112,T):new[i,0]=blend(old[i,0],settled[0],(i-112)/14)
for i in range(94,T):new[i,1]=blend(old[i,1],new[i,0]@relative,(i-94)/8)
D=new@np.linalg.inv(old)
for side in ['left','right']:
 w=np.array(row[f'robot_{side}_wrist_position']);q=np.array(row[f'robot_{side}_wrist_wxyz']);cp=np.array(row[f'mano_{side}_object_contact_positions']);cn=np.array(row[f'mano_{side}_object_contact_normals']);ids=np.array(row[f'mano_{side}_object_contact_part_ids'])
 for g in [v for v in json.loads((r/'episode-config.json').read_text())['grasps'] if v['side']==side]:
  b=g['object']
  for i in range(max(0,g['start']-4),min(T,g['end']+9)):
   weight=float(np.clip(min((i-g['start']+4)/4,(g['end']+8-i)/8),0,1));M=blend(np.eye(4),D[i,b],weight);w[i]=M[:3,:3]@w[i]+M[:3,3];q[i]=(Rotation.from_matrix(M[:3,:3])*Rotation.from_quat(q[i],scalar_first=True)).as_quat(scalar_first=True)
 for i in range(T):
  for b in range(B):
   hit=ids[i]==b+1;cp[i,hit]=cp[i,hit]@D[i,b,:3,:3].T+D[i,b,:3,3];cn[i,hit]=cn[i,hit]@D[i,b,:3,:3].T
 row[f'robot_{side}_wrist_position']=w.tolist();row[f'robot_{side}_wrist_wxyz']=q.tolist();row[f'mano_{side}_object_contact_positions']=cp.tolist();row[f'mano_{side}_object_contact_normals']=cn.tolist()
row['object_body_position']=new[:,:,:3,3].tolist();row['object_body_wxyz']=Rotation.from_matrix(new[:,:,:3,:3].reshape(-1,3,3)).as_quat(scalar_first=True).reshape(T,B,4).tolist();pq.write_table(pa.Table.from_pylist([row]),p,compression='zstd')
(r/'research/hollow-supported-stack-candidate.json').write_text(json.dumps({'diagnostic_only':True,'approved_for_training':False,'reference':ref,'method':__doc__,'fixed_stack_relative_transform_after_source_frame':102,'stack_set_down_source_frame':126,'maximum_position_change_m':np.linalg.norm(new[:,:,:3,3]-P0,axis=-1).max(axis=0).tolist(),'visual_meshes_camera_and_source_clock_unchanged':True,'gravity_evidence':'evaluation/hollow-nesting-terminal-rest-capacity2048-states.npz'},indent=2));print('10 calibrated physically settled stack and exact shared motion after inner-cup release.',flush=True)
