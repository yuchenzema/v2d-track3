"""Source-clock-preserving candidate with physically searched handover grips."""
import hashlib,json,shutil
from pathlib import Path
import numpy as np,pyarrow as pa,pyarrow.parquet as pq
from scipy.spatial.transform import Rotation,Slerp
R=Path('/home/yu_eli/v2d-track3/runs/episode_000016_newton_grasp_curriculum_20261020')
manifest=json.loads((R/'coherent-input-checksums.json').read_text())
def verify():
 for name,h in manifest.items():assert hashlib.sha256((R/name).read_bytes()).hexdigest()==h,name
verify();ref='track3_supported_handover_pair_probe';dst=R/'human_motion_data'/ref
assert not dst.exists();shutil.copytree(R/'human_motion_data/track3_fine_minimal_candidate',dst)
p=next(dst.rglob('motion.parquet'));row=pq.ParquetFile(p).read().to_pylist()[0];T=len(row['object_body_position'])
O=np.tile(np.eye(4),(T,1,1));O[:,:3,3]=np.array(row['object_body_position'])[:,1];O[:,:3,:3]=Rotation.from_quat(np.array(row['object_body_wxyz'])[:,1],scalar_first=True).as_matrix()
def smooth(v):
 v=np.clip(v,0,1);return v*v*(3-2*v)
reviews={}
for side,source in [('left',65),('right',67)]:
 report=json.loads((R/f'evaluation/handover-{side}-source{source}-consistent-search.json').read_text());assert report['robot_action_filter_initialized_at_probe_grip']
 grip=report['best_candidates']['wooden_spoon'];reviews[side]=grip
 if side=='right':assert grip['stable_hold_and_lift']
 w=np.array(row[f'robot_{side}_wrist_position']);q=np.array(row[f'robot_{side}_wrist_wxyz']);f=np.array(row[f'robot_{side}_finger_joints'])
 for i in range(T):
  if side=='left':weight=smooth((i-38)/4)*(1-smooth((i-64)/6));finger_weight=weight
  else:weight=smooth((i-60)/4)*(1-smooth((i-131)/14));finger_weight=smooth((i-60)/4)*(1-smooth((i-131)/4))
  if weight==0 and finger_weight==0:continue
  goal=O[i]@np.array(grip['wrist_in_object']);w[i]=(1-weight)*w[i]+weight*goal[:3,3]
  q[i]=Slerp([0,1],Rotation.from_matrix([Rotation.from_quat(q[i],scalar_first=True).as_matrix(),goal[:3,:3]]))([weight]).as_quat(scalar_first=True)[0]
  f[i]=(1-finger_weight)*f[i]+finger_weight*np.array(grip['finger_joints'])
 row[f'robot_{side}_wrist_position']=w.tolist();row[f'robot_{side}_wrist_wxyz']=q.tolist();row[f'robot_{side}_finger_joints']=f.tolist()
pq.write_table(pa.Table.from_pylist([row]),p,compression='zstd');verify()
out={'diagnostic_only':True,'reference':ref,'approved_for_training':False,'object_trajectories_and_source_clock_unchanged':True,
 'source_review':'research/human-handover60-70.jpg','method':__doc__,
 'left_grip_standard_lift_test_passed':reviews['left']['stable_hold_and_lift'],
 'left_grip_static_drift_m':reviews['left']['static_position_drift_m'],'left_grip_note':'Static support improved; standard isolated lift remains limited by the other hand and is not claimed to pass.',
 'right_grip_standard_lift_test_passed':True,'right_takeover_blend_source_frames':[60,64],
 'human_contact_annotations':[{'side':'right','object':1,'start':27,'end':134},{'side':'left','object':1,'start':41,'end':64}],
 'frozen_training_inputs_unchanged':True,'fk_pending':True}
(R/'research/supported-handover-pair-probe.json').write_text(json.dumps(out,indent=2));print(json.dumps(out))
