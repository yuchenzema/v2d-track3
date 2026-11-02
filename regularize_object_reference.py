"""Remove isolated nonphysical pose jumps while preserving observed timing and support anchors."""
from pathlib import Path
import argparse,json
import numpy as np,pyarrow as pa,pyarrow.parquet as pq,trimesh
from scipy.spatial.transform import Rotation,Slerp

p=argparse.ArgumentParser();p.add_argument('--run',required=True,type=Path);p.add_argument('--reference',required=True);a=p.parse_args();r=a.run;c=json.loads((r/'episode-config.json').read_text());assert a.reference!=c['training_reference'],'Write a separate reviewed candidate';path=next((r/f'human_motion_data/{a.reference}/processed').rglob('motion.parquet'));row=pq.ParquetFile(path).read().to_pylist()[0];P0=np.array(row['object_body_position']);Q0=np.array(row['object_body_wxyz']);P=P0.copy();Q=Q0.copy();T,B=P.shape[:2];records=[];changed=set()
for b,name in enumerate(c['objects']):
 R=Rotation.from_quat(Q[:,b],scalar_first=True);v=np.asarray(trimesh.load(r/f'reconstruction/mesh_priors/{name}.ply',process=False).vertices);center=v.mean(0);C=R.apply(np.tile(center,(T,1)))+P[:,b];rotstep=(R[:-1].inv()*R[1:]).magnitude();posstep=np.linalg.norm(np.diff(C,axis=0),axis=1);bad=np.flatnonzero((rotstep>.7)|(posstep>.075))+1;variable=np.zeros(T,dtype=bool)
 for frame in bad:variable[max(1,frame-5):min(T-1,frame+5)]=True
 first=min((g['start'] for g in c['grasps'] if g['object']==b),default=T);variable[:first]=False
 if c['episode']==15 and name=='wooden_spoon':
  # The video shows a stationary spoon after release. Independent framewise
  # depth/orientation fits were causing artificial motion in this interval.
  C[110:]=C[-1];q=R.as_quat();q[110:]=q[-1];R=Rotation.from_quat(q);variable[90:110]=True;variable[110:]=False
 if c['episode']==16:variable[128:]=False
 if c['episode']==30:variable[240:]=False
 if not variable.any():continue
 matrices=R.as_matrix()
 for iteration in range(160):
  for order in [range(1,T-1),range(T-2,0,-1)]:
   for i in order:
    if not variable[i]:continue
    for neighbor in [i-1,i+1]:
     delta=C[i]-C[neighbor];length=np.linalg.norm(delta)
     if length>.06:C[i]=C[neighbor]+delta*.06/length
     relative=Rotation.from_matrix(matrices[neighbor].T@matrices[i]);angle=float(relative.magnitude())
     if angle>.6:matrices[i]=matrices[neighbor]@Rotation.from_rotvec(relative.as_rotvec()*(.6/angle)).as_matrix()
 R=Rotation.from_matrix(matrices);P[:,b]=C-R.apply(np.tile(center,(T,1)));Q[:,b]=R.as_quat(scalar_first=True);after_rot=(R[:-1].inv()*R[1:]).magnitude();after_pos=np.linalg.norm(np.diff(C,axis=0),axis=1);actual=np.flatnonzero((np.linalg.norm(P[:,b]-P0[:,b],axis=1)>1e-8)|((Rotation.from_quat(Q0[:,b],scalar_first=True).inv()*R).magnitude()>1e-8));changed.update(actual.tolist());records.append({'object_index':b,'object':name,'initial_bad_frames':bad.tolist(),'changed_frames':actual.tolist(),'maximum_centroid_step_before_m':float(posstep.max()),'maximum_centroid_step_after_m':float(after_pos.max()),'maximum_rotation_step_before_rad':float(rotstep.max()),'maximum_rotation_step_after_rad':float(after_rot.max()),'stationary_after_release_enforced':c['episode']==15 and name=='wooden_spoon'})
oldrot=Rotation.from_quat(Q0.reshape(-1,4),scalar_first=True).as_matrix().reshape(T,B,3,3);newrot=Rotation.from_quat(Q.reshape(-1,4),scalar_first=True).as_matrix().reshape(T,B,3,3);D=newrot@oldrot.transpose(0,1,3,2);dt=P-np.einsum('tbij,tbj->tbi',D,P0);row['object_body_position']=P.tolist();row['object_body_wxyz']=Q.tolist()
for side in ['left','right']:
 w=np.array(row[f'robot_{side}_wrist_position']);q=np.array(row[f'robot_{side}_wrist_wxyz']);frames=np.array(row[f'robot_{side}_frames']);cp=np.array(row[f'mano_{side}_object_contact_positions']);cn=np.array(row[f'mano_{side}_object_contact_normals']);pid=np.array(row[f'mano_{side}_object_contact_part_ids'])
 for i in range(T):
  held=[g for g in c['grasps'] if g['side']==side and g['start']<=i<=g['end']];recent=[g for g in c['grasps'] if g['side']==side and g['end']<i<=g['end']+10];active=held[-1] if held else (recent[-1] if recent else None)
  if active is not None:
   b=active['object'];weight=1. if held else 1-(i-active['end'])/11;dr=Rotation.from_rotvec(Rotation.from_matrix(D[i,b]).as_rotvec()*weight);oldw=w[i].copy();neww=(1-weight)*oldw+weight*(D[i,b]@oldw+dt[i,b]);frames[i,:,:3]=dr.apply(frames[i,:,:3]-oldw)+neww;frames[i,:,3:]=(dr*Rotation.from_quat(frames[i,:,3:],scalar_first=True)).as_quat(scalar_first=True);w[i]=neww;q[i]=(dr*Rotation.from_quat(q[i],scalar_first=True)).as_quat(scalar_first=True)
  for b in range(B):hit=pid[i]==b+1;cp[i,hit]=cp[i,hit]@D[i,b].T+dt[i,b];cn[i,hit]=cn[i,hit]@D[i,b].T
 for key,value in [('wrist_position',w),('wrist_wxyz',q),('frames',frames)]:row[f'robot_{side}_{key}']=value.tolist()
 row[f'mano_{side}_object_contact_positions']=cp.tolist();row[f'mano_{side}_object_contact_normals']=cn.tolist()
pq.write_table(pa.Table.from_pylist([row]),path,compression='zstd');report={'method':__doc__,'reference':a.reference,'original_source_frame_grid_preserved':True,'original_camera_and_raw_video_reconstruction_preserved':True,'changed_frames':sorted(changed),'objects':records};(r/'research/object-reference-temporal-candidate.json').write_text(json.dumps(report,indent=2));print(c['episode'],records,flush=True)
