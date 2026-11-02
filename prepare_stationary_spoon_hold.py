"""Reconstruct the seated hold and selective finger release visible in frames95..135."""
import hashlib,json,shutil
from pathlib import Path
import numpy as np,pyarrow as pa,pyarrow.parquet as pq
from scipy.spatial.transform import Rotation,Slerp
R=Path('/home/yu_eli/v2d-track3/runs/episode_000016_newton_grasp_curriculum_20261020');frozen=json.loads((R/'coherent-input-checksums.json').read_text())
def verify():
 for p,h in frozen.items():assert hashlib.sha256((R/p).read_bytes()).hexdigest()==h,p
verify();ref='track3_stationary_spoon_hold_probe';dst=R/'human_motion_data'/ref;assert not dst.exists();shutil.copytree(R/'human_motion_data/track3_fine_minimal_candidate',dst)
p=next(dst.rglob('motion.parquet'));a=pq.ParquetFile(p).read().to_pylist()[0];P=np.array(a['object_body_position']);Q=np.array(a['object_body_wxyz']);T=len(P)
oldP=P.copy();oldQ=Q.copy();W=np.array(a['robot_right_wrist_position']);H=np.array(a['robot_right_wrist_wxyz']);F=np.array(a['robot_right_finger_joints']);oldW=W.copy();oldH=H.copy();oldF=F.copy()
def smooth(x):
 x=float(np.clip(x,0,1));return x*x*(3-2*x)
def quat(a,b,w):return Slerp([0,1],Rotation.from_quat([a,b],scalar_first=True))([w]).as_quat(scalar_first=True)[0]
# The already gravity-tested terminal spoon pose becomes the seated hold target.
for i in range(78,T):
 w=smooth((i-78)/17);P[i,1]=(1-w)*oldP[i,1]+w*oldP[-1,1];Q[i,1]=quat(oldQ[i,1],oldQ[-1,1],w)
for i in range(78,T):
 w=smooth((i-78)/22)*(1-smooth((i-131)/14));W[i]=(1-w)*oldW[i]+w*oldW[128];H[i]=quat(oldH[i],oldH[128],w)
 names=a['right_robot_finger_joint_names'];pinch=np.array([n.startswith(('right_thumb_','right_index_')) for n in names]);others=~pinch
 F[i,others]=(1-smooth((i-90)/10))*oldF[i,others]
 w=smooth((i-94)/6);F[i,pinch]=(1-w)*oldF[i,pinch]+w*oldF[128,pinch]
 F[i]*=1-smooth((i-131)/4)
a['object_body_position']=P.tolist();a['object_body_wxyz']=Q.tolist();a['robot_right_wrist_position']=W.tolist();a['robot_right_wrist_wxyz']=H.tolist();a['robot_right_finger_joints']=F.tolist();pq.write_table(pa.Table.from_pylist([a]),p,compression='zstd');verify()
report={'diagnostic_only':True,'reference':ref,'approved_for_training':False,'method':__doc__,
 'visual_evidence':'research/human-insertion78-135.jpg','object_insertion_blend_source_frames':[78,95],
 'spoon_stationary_source_frames':[95,172],'right_wrist_hold_anchor_source_frame':128,'right_wrist_blend_source_frames':[78,100],
 'nonpinching_fingers_open_source_frames':[90,100],'thumb_index_release_source_frames':[131,135],
 'terminal_support_evidence':'evaluation/fine-minimal-terminal-rest.json','rack_motion_visual_meshes_camera_and_source_clock_unchanged':True,
 'old_held_object_max_translation_to_final_m':float(np.linalg.norm(oldP[95:132,1]-oldP[-1,1],axis=1).max()),
 'old_held_object_max_rotation_to_final_rad':float((Rotation.from_quat(oldQ[95:132,1],scalar_first=True).inv()*Rotation.from_quat(oldQ[-1,1],scalar_first=True)).magnitude().max()),
 'fk_and_collision_audit_pending':True,'frozen_training_inputs_unchanged':True}
(R/'research/stationary-spoon-hold-probe.json').write_text(json.dumps(report,indent=2));print(json.dumps(report))
