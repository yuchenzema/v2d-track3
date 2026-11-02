"""Test minimal centering and axis alignment during the source-observed insertion.

Source images constrain projected silhouettes more strongly than hidden cavity
clearance. This isolated reference resolves that ambiguity with fitted cup axes;
it does not prescribe or edit policy rollout object states.
"""
import json,shutil
from pathlib import Path
import numpy as np,pyarrow as pa,pyarrow.parquet as pq
from scipy.spatial.transform import Rotation
R=Path('/home/yu_eli/v2d-track3/runs/episode_000010_newton_continuous_shell_probe_20261020')
ref='track3_axis_aligned_nesting_probe';dst=R/'human_motion_data'/ref;assert not dst.exists()
shutil.copytree(R/'human_motion_data/track3_hollow_supported_candidate',dst)
p=next(dst.rglob('motion.parquet'));row=pq.ParquetFile(p).read().to_pylist()[0]
P=np.array(row['object_body_position']);Q=np.array(row['object_body_wxyz']);oldP=P.copy();oldQ=Q.copy();T=len(P)
fit=json.loads((R.parent/'episode_000010_newton_grasp_curriculum_20261020/research/hollow-cup-collision-candidate.json').read_text())['objects']
origins=[];axes=[]
for name in ['blue_cup','beige_cup']:
 a=fit[name]['fitted_frustum_parameters'];origins.append(np.array([a[0],a[1],0.]));v=np.array([a[2],a[3],-1.]);axes.append(v/np.linalg.norm(v))
def smooth(x):
 x=np.clip(x,0,1);return x*x*(3-2*x)
deltas=[]
for i in range(T):
 weight=float(smooth((i-50)/14)*(1-smooth((i-94)/8)))
 A=Rotation.from_quat(Q[i,0],scalar_first=True);B=Rotation.from_quat(Q[i,1],scalar_first=True)
 u=A.apply(axes[0]);v=B.apply(axes[1]);cross=np.cross(v,u);angle=np.arctan2(np.linalg.norm(cross),np.dot(v,u))
 adjust=Rotation.from_rotvec(cross/max(np.linalg.norm(cross),1e-12)*angle*weight);new=adjust*B
 a=A.apply(origins[0])+P[i,0];b=B.apply(origins[1])+P[i,1];projection=a+np.dot(b-a,u)*u
 center=(1-weight)*b+weight*projection;P[i,1]=center-new.apply(origins[1]);Q[i,1]=new.as_quat(scalar_first=True)
 D=np.eye(4);D[:3,:3]=adjust.as_matrix();D[:3,3]=P[i,1]-adjust.apply(oldP[i,1]);deltas.append(D)
W=np.array(row['robot_right_wrist_position']);H=np.array(row['robot_right_wrist_wxyz'])
for i,D in enumerate(deltas):
 W[i]=D[:3,:3]@W[i]+D[:3,3];H[i]=(Rotation.from_matrix(D[:3,:3])*Rotation.from_quat(H[i],scalar_first=True)).as_quat(scalar_first=True)
row['object_body_position']=P.tolist();row['object_body_wxyz']=Q.tolist();row['robot_right_wrist_position']=W.tolist();row['robot_right_wrist_wxyz']=H.tolist()
pq.write_table(pa.Table.from_pylist([row]),p,compression='zstd')
report={'diagnostic_only':True,'approved_for_training':False,'reference':ref,'alignment_blend_source_frames':[50,64],
 'return_to_original_supported_stack_source_frames':[94,102],
 'maximum_beige_translation_m':float(np.linalg.norm(P[:,1]-oldP[:,1],axis=1).max()),
 'maximum_beige_rotation_rad':float((Rotation.from_quat(Q[:,1],scalar_first=True)*Rotation.from_quat(oldQ[:,1],scalar_first=True).inv()).magnitude().max()),
 'blue_cup_camera_clock_and_visuals_unchanged':True,'right_grasp_relative_transform_preserved':True,
 'fk_collision_and_pixel_checks_pending':True}
(R/'research/axis-aligned-nesting-probe.json').write_text(json.dumps(report,indent=2));print(report)
