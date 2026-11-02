"""Blend a source-tested, gravity-supported final pose into the physical reference.

The original camera reconstruction remains available. Only a separate candidate
reference is written; the host must audit source fidelity and collisions to accept it.
"""
from pathlib import Path
import argparse,json
import numpy as np,pyarrow as pa,pyarrow.parquet as pq
from scipy.spatial.transform import Rotation,Slerp

p=argparse.ArgumentParser();p.add_argument('--run',required=True,type=Path);p.add_argument('--reference',required=True);a=p.parse_args();r=a.run;c=json.loads((r/'episode-config.json').read_text());assert not (r/'coherent-input-checksums.json').exists();search=json.loads((r/'research/terminal-support-search.json').read_text());best=search['best'];assert best is not None
path=next((r/f'human_motion_data/{a.reference}/processed').rglob('motion.parquet'));row=pq.ParquetFile(path).read().to_pylist()[0];P0=np.array(row['object_body_position']);Q0=np.array(row['object_body_wxyz']);P=P0.copy();Q=Q0.copy();b=search['object_index'];T=c['frames'];B=len(c['objects']);start,end={14:(80,108),16:(98,128),20:(132,159),28:(146,190),30:(212,240)}[c['episode']];pose=np.array(best['settled_pose']);target_rotation=Rotation.from_quat(pose[3:]);
for i in range(start,T):
 blend=float(np.clip((i-start)/max(1,end-start),0,1));blend=blend*blend*(3-2*blend)
 target_position=pose[:3];frame_rotation=target_rotation
 if c['episode']==30:
  # The wearer still moves the rack during insertion. A supported brush must
  # follow that observed rack motion instead of freezing in world space early.
  support=c['objects'].index('plastic_dish_rack');relative=Rotation.from_quat(Q0[i,support],scalar_first=True)*Rotation.from_quat(Q0[-1,support],scalar_first=True).inv();target_position=P0[i,support]+relative.apply(pose[:3]-P0[-1,support]);frame_rotation=relative*target_rotation
 P[i,b]=(1-blend)*P0[i,b]+blend*target_position;Q[i,b]=Slerp([0,1],Rotation.from_quat([Q0[i,b],frame_rotation.as_quat(scalar_first=True)],scalar_first=True))([blend]).as_quat(scalar_first=True)[0]
old_rot=Rotation.from_quat(Q0.reshape(-1,4),scalar_first=True).as_matrix().reshape(T,B,3,3);new_rot=Rotation.from_quat(Q.reshape(-1,4),scalar_first=True).as_matrix().reshape(T,B,3,3);D=new_rot@old_rot.transpose(0,1,3,2);dt=P-np.einsum('tbij,tbj->tbi',D,P0);row['object_body_position']=P.tolist();row['object_body_wxyz']=Q.tolist()
for side in ['left','right']:
 w=np.array(row[f'robot_{side}_wrist_position']);wq=np.array(row[f'robot_{side}_wrist_wxyz']);frames=np.array(row[f'robot_{side}_frames']);cp=np.array(row[f'mano_{side}_object_contact_positions']);cn=np.array(row[f'mano_{side}_object_contact_normals']);pid=np.array(row[f'mano_{side}_object_contact_part_ids'])
 for i in range(T):
  held=any(g['side']==side and g['object']==b and g['start']<=i<=g['end'] for g in c['grasps']);recent=[g for g in c['grasps'] if g['side']==side and g['object']==b and g['end']<i<=g['end']+10]
  weight=1. if held else (max(1-(i-g['end'])/11 for g in recent) if recent else 0.)
  if weight>0:
   dr=Rotation.from_rotvec(Rotation.from_matrix(D[i,b]).as_rotvec()*weight).as_matrix();shift=dt[i,b]*weight;w[i]=dr@w[i]+shift;wq[i]=(Rotation.from_matrix(dr)*Rotation.from_quat(wq[i],scalar_first=True)).as_quat(scalar_first=True);frames[i,:,:3]=frames[i,:,:3]@dr.T+shift;frames[i,:,3:]=(Rotation.from_matrix(dr)*Rotation.from_quat(frames[i,:,3:],scalar_first=True)).as_quat(scalar_first=True)
  hit=pid[i]==b+1;cp[i,hit]=cp[i,hit]@D[i,b].T+dt[i,b];cn[i,hit]=cn[i,hit]@D[i,b].T
 for key,value in [('wrist_position',w),('wrist_wxyz',wq),('frames',frames)]:row[f'robot_{side}_{key}']=value.tolist()
 row[f'mano_{side}_object_contact_positions']=cp.tolist();row[f'mano_{side}_object_contact_normals']=cn.tolist()
pq.write_table(pa.Table.from_pylist([row]),path,compression='zstd');report={'method':__doc__,'candidate_reference':a.reference,'object':c['objects'][b],'blend_interval':[start,end],'supported_pose':best,'original_video_reconstruction_preserved':True,'original_frame_grid_preserved':True,'maximum_object_position_change_m':float(np.linalg.norm(P-P0,axis=-1).max()),'maximum_object_rotation_change_rad':float(Rotation.from_matrix(D.reshape(-1,3,3)).magnitude().max())};(r/'research/terminal-support-reference-candidate.json').write_text(json.dumps(report,indent=2));print(c['episode'],report,flush=True)
