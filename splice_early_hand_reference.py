"""Replace the incorrect unobserved approach, preserving all fitted sustained grasps."""
from pathlib import Path
import json,os
import numpy as np,pyarrow as pa,pyarrow.parquet as pq,trimesh
from scipy.spatial import cKDTree
from scipy.spatial.transform import Rotation,Slerp
from robotic_grounding.retarget.retarget_utils import setup_sharpa_kinematics
r=Path('/episode');assert not (r/'coherent-input-checksums.json').exists();c=json.loads((r/'episode-config.json').read_text());assert c['episode']==19;reference=c.get('training_reference','track3_collision_variant');fresh=pq.ParquetFile(next((r/'human_motion_data/track3_early_hand_repair_raw').rglob('motion.parquet'))).read().to_pylist()[0];source_names=fresh['left_robot_frame_names'];kin=setup_sharpa_kinematics('left',frequency=50.,max_iter=100,frame_tasks_converged_threshold=2e-6);meshes=[trimesh.load(r/f'reconstruction/mesh_priors/{n}.ply',process=False) for n in c['objects']];trees=[cKDTree(mesh.vertices) for mesh in meshes];reports=[]
for ref,end in [(reference,7),('track3_video',28)]:
 path=next((r/f'human_motion_data/{ref}').rglob('motion.parquet'));row=pq.ParquetFile(path).read().to_pylist()[0];old={key:np.array(row[key]) for key in ['object_body_position','object_body_wxyz','robot_right_wrist_position','robot_right_wrist_wxyz','robot_right_finger_joints','robot_right_frames','robot_left_wrist_position','robot_left_wrist_wxyz','robot_left_finger_joints','robot_left_frames']};w=np.array(row['robot_left_wrist_position']);q=np.array(row['robot_left_wrist_wxyz']);f=np.array(row['robot_left_finger_joints']);frames=np.array(row['robot_left_frames']);assert row['left_robot_frame_names']==source_names
 for key,array in [('wrist_position',w),('wrist_wxyz',q),('finger_joints',f),('frames',frames)]:array[:end]=np.array(fresh[f'robot_left_{key}'])[:end]
 if ref==reference:
  # Meet the already collision-fitted grasp at frame 7 smoothly. The grasp
  # itself and the other robot's entire trajectory remain exactly unchanged.
  for i in range(3,7):
   blend=((i-2)/5)**2*(3-2*(i-2)/5);w[i]=(1-blend)*w[i]+blend*old['robot_left_wrist_position'][7];q[i]=Slerp([0,1],Rotation.from_quat([q[i],old['robot_left_wrist_wxyz'][7]],scalar_first=True))([blend]).as_quat(scalar_first=True)[0];f[i]=(1-blend)*f[i]+blend*old['robot_left_finger_joints'][7]
  positions=np.r_[w[:7],old['robot_left_wrist_position'][7:8]]
  for repeat in range(100):
   for i in range(1,7):
    delta=positions[i]-positions[i-1];length=np.linalg.norm(delta)
    if length>.05:positions[i]=positions[i-1]+delta*.05/length
   for i in range(6,0,-1):
    delta=positions[i]-positions[i+1];length=np.linalg.norm(delta)
    if length>.05:positions[i]=positions[i+1]+delta*.05/length
  w[:7]=positions[:7]
 for i in range(end):
  kin.configuration.update(np.r_[w[i],q[i,[1,2,3,0]],f[i]])
  for j,name in enumerate(source_names):
   transform=kin.configuration.get_transform_frame_to_world(name);frames[i,j]=np.r_[transform.translation,Rotation.from_matrix(transform.rotation).as_quat(scalar_first=True)]
 for key,value in [('wrist_position',w),('wrist_wxyz',q),('finger_joints',f),('frames',frames)]:row[f'robot_left_{key}']=value.tolist()
 P=np.array(row['object_body_position']);Q=np.array(row['object_body_wxyz']);cp=np.array(row['mano_left_object_contact_positions']);cn=np.array(row['mano_left_object_contact_normals']);pid=np.array(row['mano_left_object_contact_part_ids']);tips=[source_names.index(f'left_{name}_tip_site') for name in ['thumb','index','middle','ring','pinky']]
 for i in range(end):
  cp[i]=0;cn[i]=0;pid[i]=0;active=[g['object'] for g in c['grasps'] if g['side']=='left' and max(0,g['start']-3)<=i<=g['end']]
  for j,tip in enumerate(tips):
   choices=[]
   for b in active:
    R=Rotation.from_quat(Q[i,b],scalar_first=True).as_matrix();distance,index=trees[b].query((frames[i,tip,:3]-P[i,b])@R);choices.append((distance,b,index))
   if not choices:continue
   distance,b,index=min(choices)
   if distance>.025:continue
   R=Rotation.from_quat(Q[i,b],scalar_first=True).as_matrix();cp[i,j]=R@meshes[b].vertices[index]+P[i,b];normal=R@meshes[b].vertex_normals[index];cn[i,j]=normal/(np.linalg.norm(normal)+1e-12);pid[i,j]=b+1
 for key,value in [('positions',cp),('normals',cn),('part_ids',pid)]:row[f'mano_left_object_contact_{key}']=value.tolist()
 for key in old:
  if not key.startswith('robot_left'):assert np.array_equal(old[key],np.array(row[key])),key
 if ref==reference:
  for key in old:
   if key.startswith('robot_left'):assert np.array_equal(old[key][7:],np.array(row[key])[7:]),key
 pq.write_table(pa.Table.from_pylist([row]),path,compression='zstd');owner=r.stat();os.chown(path,owner.st_uid,owner.st_gid);reports.append({'reference':ref,'changed_left_frames':[0,end-1],'all_objects_and_entire_right_hand_preserved':True,'sustained_grasps_preserved':ref==reference})
(r/'research/early-hand-reference-splice.json').write_text(json.dumps({'method':__doc__,'changes':reports},indent=2));print(reports,flush=True)
