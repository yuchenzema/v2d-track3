"""Join the image-fitted grasp to its existing approach and open-hand withdrawal."""
import json,os
from pathlib import Path
import numpy as np,pyarrow as pa,pyarrow.parquet as pq,trimesh
from scipy.spatial import cKDTree
from scipy.spatial.transform import Rotation,Slerp
from robotic_grounding.retarget.retarget_utils import setup_sharpa_kinematics
r=Path('/episode');assert not (r/'coherent-input-checksums.json').exists();cfg=json.loads((r/'episode-config.json').read_text());selection=json.loads((r/'research/source-aligned-grasp-selection.json').read_text());assert selection['accepted'];reference=cfg.get('training_reference','track3_collision_variant');path=next((r/f'human_motion_data/{reference}').rglob('motion.parquet'));row=pq.ParquetFile(path).read().to_pylist()[0];old=pq.ParquetFile(r/selection['backup']/'motion.parquet').read().to_pylist()[0];side=selection['affected_side'];w=np.array(row[f'robot_{side}_wrist_position']);q=np.array(row[f'robot_{side}_wrist_wxyz']);finger=np.array(row[f'robot_{side}_finger_joints']);frames=np.array(row[f'robot_{side}_frames']);ow=np.array(old[f'robot_{side}_wrist_position']);oq=np.array(old[f'robot_{side}_wrist_wxyz']);T=cfg['frames'];changed=set();intervals=[g for g in cfg['grasps'] if g['side']==side]
for g in intervals:
 for anchor,indices in [(g['start'],range(max(0,g['start']-4),g['start'])),(g['end'],range(g['end']+1,min(T,g['end']+8)))]:
  delta=Rotation.from_quat(q[anchor],scalar_first=True)*Rotation.from_quat(oq[anchor],scalar_first=True).inv()
  for i in indices:
   if any(x['start']<=i<=x['end'] for x in intervals):continue
   b=max(0.,1-abs(i-anchor)/(4 if i<anchor else 8));b=b*b*(3-2*b);target=w[anchor]+delta.apply(ow[i]-ow[anchor]);w[i]=(1-b)*ow[i]+b*target;q[i]=(Rotation.from_rotvec(delta.as_rotvec()*b)*Rotation.from_quat(oq[i],scalar_first=True)).as_quat(scalar_first=True);changed.add(i)
names=row[f'{side}_robot_frame_names'];kin=setup_sharpa_kinematics(side,frequency=50.,max_iter=100,frame_tasks_converged_threshold=2e-6);meshes=[trimesh.load(r/f'reconstruction/mesh_priors/{n}.ply',process=False) for n in cfg['objects']];trees=[cKDTree(m.vertices) for m in meshes];P=np.array(row['object_body_position']);Q=np.array(row['object_body_wxyz']);cp=np.array(row[f'mano_{side}_object_contact_positions']);cn=np.array(row[f'mano_{side}_object_contact_normals']);pid=np.array(row[f'mano_{side}_object_contact_part_ids']);tips=[names.index(f'{side}_{x}_tip_site') for x in ['thumb','index','middle','ring','pinky']]
for i in changed:
 kin.configuration.update(np.r_[w[i],q[i,[1,2,3,0]],finger[i]])
 for j,name in enumerate(names):tr=kin.configuration.get_transform_frame_to_world(name);frames[i,j]=np.r_[tr.translation,Rotation.from_matrix(tr.rotation).as_quat(scalar_first=True)]
 cp[i]=0;cn[i]=0;pid[i]=0;active=[g['object'] for g in intervals if max(0,g['start']-3)<=i<=g['end']]
 for j,tip in enumerate(tips):
  choices=[]
  for b in active:
   R=Rotation.from_quat(Q[i,b],scalar_first=True).as_matrix();distance,index=trees[b].query((frames[i,tip,:3]-P[i,b])@R);choices.append((distance,b,index))
  if not choices:continue
  distance,b,index=min(choices)
  if distance>.025:continue
  R=Rotation.from_quat(Q[i,b],scalar_first=True).as_matrix();cp[i,j]=R@meshes[b].vertices[index]+P[i,b];normal=R@meshes[b].vertex_normals[index];cn[i,j]=normal/(np.linalg.norm(normal)+1e-12);pid[i,j]=b+1
for key,value in [('wrist_position',w),('wrist_wxyz',q),('frames',frames)]:row[f'robot_{side}_{key}']=value.tolist()
for key,value in [('positions',cp),('normals',cn),('part_ids',pid)]:row[f'mano_{side}_object_contact_{key}']=value.tolist()
pq.write_table(pa.Table.from_pylist([row]),path,compression='zstd');owner=r.stat();os.chown(path,owner.st_uid,owner.st_gid);(r/'research/source-aligned-grasp-boundaries.json').write_text(json.dumps({'method':__doc__,'side':side,'changed_unheld_frames':sorted(changed),'open_hand_finger_motion_preserved':True,'all_held_poses_objects_and_camera_preserved':True},indent=2));print(side,sorted(changed),flush=True)
