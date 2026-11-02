"""Isolated handover candidate: preserve the observed side of the right-hand grip."""
from pathlib import Path
import hashlib,json,shutil
import numpy as np,pyarrow as pa,pyarrow.parquet as pq
from scipy.spatial.transform import Rotation,Slerp
R=Path('/home/yu_eli/v2d-track3/runs/episode_000016_newton_grasp_curriculum_20261020')
frozen=json.loads((R/'coherent-input-checksums.json').read_text())
def verify():
 for p,h in frozen.items():assert hashlib.sha256((R/p).read_bytes()).hexdigest()==h,p
verify()
for slide in [0.,.04]:
 ref=f'track3_continuous_right_slide{round(slide*1000):03d}_probe';dst=R/'human_motion_data'/ref
 assert not dst.exists();shutil.copytree(R/'human_motion_data/track3_fine_minimal_candidate',dst)
 p=next(dst.rglob('motion.parquet'));row=pq.ParquetFile(p).read().to_pylist()[0];T=len(row['object_body_position'])
 O=np.tile(np.eye(4),(T,1,1));O[:,:3,3]=np.array(row['object_body_position'])[:,1];O[:,:3,:3]=Rotation.from_quat(np.array(row['object_body_wxyz'])[:,1],scalar_first=True).as_matrix()
 w=np.array(row['robot_right_wrist_position']);q=np.array(row['robot_right_wrist_wxyz']);f=np.array(row['robot_right_finger_joints'])
 H=np.tile(np.eye(4),(T,1,1));H[:,:3,3]=w;H[:,:3,:3]=Rotation.from_quat(q,scalar_first=True).as_matrix();local=np.linalg.inv(O[60])@H[60];finger=f[60].copy()
 for i in range(62,145):
  progress=float(np.clip((i-61)/11,0,1));progress=progress*progress*(3-2*progress)
  goal=local.copy();goal[2,3]+=slide*progress;world=O[i]@goal
  release=float(np.clip((i-131)/13,0,1));release=release*release*(3-2*release)
  w[i]=(1-release)*world[:3,3]+release*H[i,:3,3]
  q[i]=Slerp([0,1],Rotation.from_matrix([world[:3,:3],H[i,:3,:3]]))([release]).as_quat(scalar_first=True)[0]
  # Preserve the human release finger-opening schedule while removing the artificial midair flip.
  finger_release=float(np.clip((i-131)/4,0,1));f[i]=(1-finger_release)*finger+finger_release*f[i]
 row['robot_right_wrist_position']=w.tolist();row['robot_right_wrist_wxyz']=q.tolist();row['robot_right_finger_joints']=f.tolist()
 pq.write_table(pa.Table.from_pylist([row]),p,compression='zstd')
 report={'diagnostic_only':True,'reference':ref,'approved_for_training':False,'right_grip_local_pose_source_frame':60,
  'axial_slide_m':slide,'modified_source_frames':[62,144],'object_trajectories_unchanged':True,'visual_meshes_camera_and_clock_unchanged':True,
  'reason':'Direct source frames60,62,64,65,66,67,68,70 show continuous right-hand contact. Existing reference abruptly moves the wrist about20cm to the opposite side during62..66.',
  'source_video_review':'research/human-handover60-70.jpg','fk_pending':True,'frozen_training_inputs_unchanged':True}
 (R/f'research/{ref}.json').write_text(json.dumps(report,indent=2));print(ref,flush=True)
verify()
