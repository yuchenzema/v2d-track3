"""Restore source-fitted cup poses while keeping grasped robot hands attached consistently."""
import argparse,json,shutil
from pathlib import Path
import numpy as np,pyarrow as pa,pyarrow.parquet as pq
from scipy.spatial.transform import Rotation
p=argparse.ArgumentParser();p.add_argument('--run',type=Path,required=True);p.add_argument('--reference',default='track3_source_cavity_candidate');a=p.parse_args();r=a.run
assert not (r/'coherent-input-checksums.json').exists()
c=json.loads((r/'episode-config.json').read_text());assert c['episode']==10
ref=a.reference;assert ref.replace('_','').isalnum();dst=r/'human_motion_data'/ref
shutil.copytree(r/'human_motion_data'/c['training_reference'],dst,dirs_exist_ok=True)
path=next(dst.rglob('motion.parquet'));row=pq.ParquetFile(path).read().to_pylist()[0]
raw=pq.ParquetFile(next((r/'human_motion_data/track3_video').rglob('motion.parquet'))).read().to_pylist()[0]
P0=np.array(row['object_body_position']);Q0=np.array(row['object_body_wxyz']);P=np.array(raw['object_body_position']);Q=np.array(raw['object_body_wxyz'])
T,B=P.shape[:2];old=Rotation.from_quat(Q0.reshape(-1,4),scalar_first=True).as_matrix().reshape(T,B,3,3);new=Rotation.from_quat(Q.reshape(-1,4),scalar_first=True).as_matrix().reshape(T,B,3,3);D=new@old.transpose(0,1,3,2);shift=P-np.einsum('tbij,tbj->tbi',D,P0)
for side in ['left','right']:
 w=np.array(row[f'robot_{side}_wrist_position']);q=np.array(row[f'robot_{side}_wrist_wxyz']);frames=np.array(row[f'robot_{side}_frames'])
 for g in [x for x in c['grasps'] if x['side']==side]:
  b=g['object'];start,end=g['start'],g['end']
  for i in range(max(0,start-8),min(T,end+9)):
   f=float(np.clip(min((i-start+8)/8,(end+8-i)/8),0.,1.));f=f*f*(3-2*f)
   rot=Rotation.from_rotvec(Rotation.from_matrix(D[i,b]).as_rotvec()*f);m=rot.as_matrix();delta=f*shift[i,b]
   w[i]=m@w[i]+delta;q[i]=(rot*Rotation.from_quat(q[i],scalar_first=True)).as_quat(scalar_first=True)
   frames[i,:,:3]=frames[i,:,:3]@m.T+delta;frames[i,:,3:]=(rot*Rotation.from_quat(frames[i,:,3:],scalar_first=True)).as_quat(scalar_first=True)
 cp=np.array(row[f'mano_{side}_object_contact_positions']);cn=np.array(row[f'mano_{side}_object_contact_normals']);pid=np.array(row[f'mano_{side}_object_contact_part_ids'])
 for i in range(T):
  for b in range(B):
   hit=pid[i]==b+1;cp[i,hit]=cp[i,hit]@D[i,b].T+shift[i,b];cn[i,hit]=cn[i,hit]@D[i,b].T
 for k,v in [('wrist_position',w),('wrist_wxyz',q),('frames',frames)]:row[f'robot_{side}_{k}']=v.tolist()
 row[f'mano_{side}_object_contact_positions']=cp.tolist();row[f'mano_{side}_object_contact_normals']=cn.tolist()
row['object_body_position']=P.tolist();row['object_body_wxyz']=Q.tolist();pq.write_table(pa.Table.from_pylist([row]),path,compression='zstd')
report={'diagnostic_only':True,'approved_for_training':False,'reference':ref,'source_object_poses_restored_exactly':True,
 'finger_joint_targets_unchanged':True,'camera_source_frames_and_visual_meshes_unchanged':True,
 'maximum_translation_from_prior_physical_reference_m':np.linalg.norm(P-P0,axis=-1).max(axis=0).tolist()}
output='source-cavity-reference-candidate.json' if ref=='track3_source_cavity_candidate' else ref+'-source-reference.json'
(r/'research'/output).write_text(json.dumps(report,indent=2));print(report,flush=True)
