"""Blend a previously checked insertion grip only after the learned handover."""
from pathlib import Path
import json,hashlib,shutil
import numpy as np,pyarrow as pa,pyarrow.parquet as pq
from scipy.spatial.transform import Rotation,Slerp
R=Path('/home/yu_eli/v2d-track3/runs/episode_000016_newton_grasp_curriculum_20261020')
frozen=json.loads((R/'coherent-input-checksums.json').read_text())
def verify():
 for p,h in frozen.items():assert hashlib.sha256((R/p).read_bytes()).hexdigest()==h,p
verify();ref='track3_late_high_grip_probe';dst=R/'human_motion_data'/ref;assert not dst.exists()
shutil.copytree(R/'human_motion_data/track3_fine_minimal_candidate',dst)
p=next(dst.rglob('motion.parquet'));a=pq.ParquetFile(p).read().to_pylist()[0]
b=pq.ParquetFile(next((R/'human_motion_data/track3_fine_transition_candidate').rglob('motion.parquet'))).read().to_pylist()[0]
for key in ['object_body_position','object_body_wxyz']:np.testing.assert_array_equal(a[key],b[key])
w=np.array(a['robot_right_wrist_position']);q=np.array(a['robot_right_wrist_wxyz']);f=np.array(a['robot_right_finger_joints'])
for i in range(84,len(w)):
 weight=float(np.clip((i-84)/16,0,1));weight=weight*weight*(3-2*weight)
 w[i]=(1-weight)*w[i]+weight*np.asarray(b['robot_right_wrist_position'][i])
 q[i]=Slerp([0,1],Rotation.from_quat([q[i],b['robot_right_wrist_wxyz'][i]],scalar_first=True))([weight]).as_quat(scalar_first=True)[0]
 f[i]=(1-weight)*f[i]+weight*np.asarray(b['robot_right_finger_joints'][i])
a['robot_right_wrist_position']=w.tolist();a['robot_right_wrist_wxyz']=q.tolist();a['robot_right_finger_joints']=f.tolist()
pq.write_table(pa.Table.from_pylist([a]),p,compression='zstd');verify()
(R/'research/late-high-grip-probe.json').write_text(json.dumps({'diagnostic_only':True,'reference':ref,'approved_for_training':False,
 'method':__doc__,'blend_source_frames':[84,100],'prior_grip_reference':'track3_fine_transition_candidate',
 'required_robot_impedance':'right250N/m,20Ns/m after source100, ramp beginning84; native45N actuator limits',
 'human_object_trajectory_camera_visual_meshes_clock_unchanged':True,'early_robot_reference_through_source84_unchanged':True,
 'frozen_training_inputs_unchanged':True,'fk_pending':True},indent=2));print(ref)
