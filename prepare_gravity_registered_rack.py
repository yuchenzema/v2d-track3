"""Create an isolated rack/holder registration probe without changing training inputs."""
from pathlib import Path
import hashlib,json,shutil
import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq
from scipy.spatial.transform import Rotation,Slerp

R=Path('/home/yu_eli/v2d-track3/runs/episode_000016_newton_grasp_curriculum_20261020')
manifest=json.loads((R/'coherent-input-checksums.json').read_text())
def verify():
 for name,h in manifest.items():assert hashlib.sha256((R/name).read_bytes()).hexdigest()==h,name
verify()
reference='track3_gravity_supported_probe';src=R/'human_motion_data/track3_fine_minimal_candidate';dst=R/'human_motion_data'/reference
assert not dst.exists(),'Keep previous diagnostic references immutable'
shutil.copytree(src,dst)
p=next(dst.rglob('motion.parquet'));row=pq.ParquetFile(p).read().to_pylist()[0]
P=np.array(row['object_body_position']);Q=np.array(row['object_body_wxyz']);T,B=P.shape[:2]
old=np.tile(np.eye(4),(T,B,1,1));old[:,:,:3,3]=P;old[:,:,:3,:3]=Rotation.from_quat(Q.reshape(-1,4),scalar_first=True).as_matrix().reshape(T,B,3,3)
new=old.copy()
pose=np.array(json.loads((R/'research/rack-support-registration-audit.json').read_text())['initial_gravity_settled_rack_pose_wxyz'])
rack=np.eye(4);rack[:3,3]=pose[:3];rack[:3,:3]=Rotation.from_quat(pose[3:],scalar_first=True).as_matrix()
def blend(A,B,w):
 w=float(np.clip(w,0,1));w=w*w*(3-2*w);M=np.eye(4);M[:3,3]=(1-w)*A[:3,3]+w*B[:3,3];M[:3,:3]=Slerp([0,1],Rotation.from_matrix([A[:3,:3],B[:3,:3]]))([w]).as_matrix()[0];return M
# Keep the observed pickup and regrasp intact; register the shaft to the resting holder.
for i in range(T):
 new[i,0]=rack
 new[i,1]=blend(old[i,1],rack@np.linalg.inv(old[i,0])@old[i,1],(i-67)/12)
D=new@np.linalg.inv(old)
cfg=json.loads((R/'episode-config.json').read_text())
for side in ['left','right']:
 w=np.array(row[f'robot_{side}_wrist_position']);q=np.array(row[f'robot_{side}_wrist_wxyz'])
 for g in [g for g in cfg['grasps'] if g['side']==side]:
  for i in range(max(0,g['start']-4),min(T,g['end']+9)):
   weight=np.clip(min((i-g['start']+4)/4,(g['end']+8-i)/8),0,1)
   M=blend(np.eye(4),D[i,g['object']],weight);w[i]=M[:3,:3]@w[i]+M[:3,3];q[i]=(Rotation.from_matrix(M[:3,:3])*Rotation.from_quat(q[i],scalar_first=True)).as_quat(scalar_first=True)
 row[f'robot_{side}_wrist_position']=w.tolist();row[f'robot_{side}_wrist_wxyz']=q.tolist()
row['object_body_position']=new[:,:,:3,3].tolist();row['object_body_wxyz']=Rotation.from_matrix(new[:,:,:3,:3].reshape(-1,3,3)).as_quat(scalar_first=True).reshape(T,B,4).tolist()
pq.write_table(pa.Table.from_pylist([row]),p,compression='zstd');verify()
report={'diagnostic_only':True,'reference':reference,'approved_for_training':False,'frozen_inputs_unchanged':True,
 'source_human_video_used':True,'rack_pose_source':'research/rack-support-registration-audit.json',
 'spoon_registration_blend_source_frames':[67,79],'maximum_object_translation_change_m':np.linalg.norm(new[:,:,:3,3]-P,axis=-1).max(axis=0).tolist(),
 'fk_and_collision_audit_pending':True,'evaluation_object_assistance':0.}
(R/'research/gravity-registered-rack-probe.json').write_text(json.dumps(report,indent=2));print(json.dumps(report))
