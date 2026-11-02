"""Fit the second video-observed grasp above the rack while retaining the supported spoon path."""
import ast,json,os,shlex,shutil,subprocess
from pathlib import Path
import numpy as np,pyarrow as pa,pyarrow.parquet as pq
from scipy.spatial.transform import Rotation
from improvement_common import BASE,save,utc,verify_preserved
r=BASE/'runs/episode_000016_newton_grasp_curriculum_20261020';assert not (r/'coherent-input-checksums.json').exists()
rows=[]
for line in (r/'logs/high-insertion-grasp-search.log').read_text().splitlines():
 if line.startswith('wooden_spoon generation ') and ' best {' in line:
  x=ast.literal_eval('{'+line.split(' best {',1)[1])
  if x['stable_hold_and_lift'] and x['other_objects_max_translation_m']<=.008 and x['other_objects_max_orientation_rad']<=.12:rows.append(x)
assert rows,'No grasp protects the stationary rack adequately'
best=min(rows,key=lambda x:x['cost']);ref='track3_fine_clearance_candidate';original=r/'human_motion_data/track3_fine_insertion_candidate';candidate=r/'human_motion_data'/ref
shutil.copytree(original,candidate,dirs_exist_ok=True);path=next(candidate.rglob('motion.parquet'));old=pq.ParquetFile(next(original.rglob('motion.parquet'))).read().to_pylist()[0]
fit={'side':'right','object_index':1,'object':'wooden_spoon','held_interval':[66,134],'wrist_in_object':best['wrist_in_object'],
 'finger_joint_names':best['finger_joint_names'],'finger_joints':best['finger_joints'],'template_source_frame':103}
save(r/'research/fine-clearance-grasp-fit.json',{'grasps':[fit],'chosen_physics_probe':best,
 'reason':'Prefer the feasible grip that keeps rack movement below8mm and rotation below0.12rad.'})
shutil.copy2(BASE/'apply_collision_transitions.py',r/'scripts/apply_collision_transitions.py')
image=subprocess.check_output(['sg','docker','-c',shlex.join(['docker','inspect','v2d-batch-episode32-retarget','--format','{{.Image}}'])],text=True).strip()
def fk(label,rebuild=False):
 command=['docker','run','--rm','--name','v2d-spoon16-clearance-fk-cpu','--ipc','host','-e','CUDA_VISIBLE_DEVICES=',
  '-e',f'V2D_TRANSITION_REFERENCE={ref}','-e','V2D_CONTACT_FIT_OUTPUT=research/fine-clearance-grasp-fit.json',
  '-e',f'V2D_CONTACT_APPLY_OUTPUT=research/{label}.json','-e',f'V2D_REBUILD_ONLY={int(rebuild)}',
  '-v',f'{r}:/episode','-v',f'{BASE}/video_to_data/robotic_grounding/source/robotic_grounding/robotic_grounding/assets:/v2d/assets:ro',
  '--entrypoint','/isaac-sim/python.sh',image,'-u','/episode/scripts/apply_collision_transitions.py']
 with (r/f'logs/{label}.log').open('w') as f:subprocess.run(['sg','docker','-c',shlex.join(command)],stdout=f,stderr=subprocess.STDOUT,check=True)
fk('fine-clearance-applied')
row=pq.ParquetFile(path).read().to_pylist()[0];w=np.array(row['robot_right_wrist_position']);q=np.array(row['robot_right_wrist_wxyz']);f=np.array(row['robot_right_finger_joints'])
ow=np.array(old['robot_right_wrist_position']);oq=np.array(old['robot_right_wrist_wxyz']);of=np.array(old['robot_right_finger_joints'])
for anchor,indices,width in [(66,range(62,66),4),(134,range(135,143),8)]:
 delta=Rotation.from_quat(q[anchor],scalar_first=True)*Rotation.from_quat(oq[anchor],scalar_first=True).inv()
 for i in indices:
  b=np.clip(1-abs(i-anchor)/width,0.,1.);b=b*b*(3-2*b);w[i]=(1-b)*ow[i]+b*(w[anchor]+delta.apply(ow[i]-ow[anchor]))
  q[i]=(Rotation.from_rotvec(delta.as_rotvec()*b)*Rotation.from_quat(oq[i],scalar_first=True)).as_quat(scalar_first=True)
  if i<anchor:f[i]=(1-b)*of[i]+b*f[anchor]
held=f[131].copy();opened=of[140].copy()
for i in range(132,141):
 b=float(np.clip((i-132)/3,0.,1.));b=b*b*(3-2*b);f[i]=(1-b)*held+b*opened
for k,v in [('wrist_position',w),('wrist_wxyz',q),('finger_joints',f)]:row['robot_right_'+k]=v.tolist()
pq.write_table(pa.Table.from_pylist([row]),path,compression='zstd');fk('fine-clearance-rebuilt-fields',rebuild=True)
row=pq.ParquetFile(path).read().to_pylist()[0]
for k in ['object_body_position','object_body_wxyz','object_mesh_paths','object_urdf_paths']:assert np.array_equal(row[k],old[k])
save(r/'research/fine-clearance-candidate.json',{'updated_utc':utc(),'reference':ref,'diagnostic_only':True,'approved_for_training':False,
 'object_poses_and_camera_unchanged_from_feasible_insertion_candidate':True,'robot_second_grasp_interval':[66,134],
 'robot_pregrasp_blend_interval':[62,66],'robot_finger_release_blend_interval':[132,135],'observed_release_frame':134,
 'physics_template_evidence':'research/fine-clearance-grasp-fit.json'})
verify_preserved(16);print('16 clearance candidate rebuilt with fixed object motion and explicit release timing.',flush=True)
