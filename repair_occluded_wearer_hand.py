"""Remove episode 16's bystander-hand track during the wearer's left-hand occlusion.

The missing interval is explicitly inferred outside the camera view, with smooth
exit/reentry. No held grasp, object pose, camera pose or other-hand pose is changed.
"""
import json,os
from pathlib import Path
import numpy as np,pyarrow as pa,pyarrow.parquet as pq
from scipy.spatial.transform import Rotation,Slerp
from robotic_grounding.retarget.retarget_utils import setup_sharpa_kinematics

r=Path('/episode');c=json.loads((r/'episode-config.json').read_text());assert c['episode']==16 and not (r/'coherent-input-checksums.json').exists();lo,hi=81,139;ids=range(lo+1,hi);assert not any(g['side']=='left' and g['start']<=hi-1 and g['end']>=lo+1 for g in c['grasps']);rec=r/'reconstruction';motion={k:v.copy() for k,v in np.load(rec/'scene_motion_physical.npz').items()};camera=motion['camera_to_world'];intr=json.loads((rec/'camera_intrinsics.json').read_text());K=np.array([[intr['fx'],0,intr['cx']],[0,intr['fy'],intr['cy']],[0,0,1.]])
def camera_points(points,i):
 C=np.linalg.inv(camera[i]);return points@C[:3,:3].T+C[:3,3]
def missing_path(points):
 a=camera_points(points[lo],lo);b=camera_points(points[hi],hi);out=points.copy()
 for i in ids:
  t=(i-lo)/(hi-lo);base=(1-t)*a+t*b;uv=(K@base)[:2]/base[2];outside=np.linalg.solve(K,[uv[0],980.,1.])*base[2];weight=min(1.,(i-lo)/5,(hi-i)/5);weight=weight*weight*(3-2*weight);pc=(1-weight)*base+weight*outside;out[i]=camera[i,:3,:3]@pc+camera[i,:3,3]
 return out

measure={k:v.copy() for k,v in np.load(rec/'hand_measurements.npz').items()};bad=[i for i in range(95,130) if measure['observed'][i,0]]
for i in bad:
 measure['observed'][i,0]=False;measure['uv'][i,0]=np.nan;measure['relative_xyz'][i,0]=np.nan;measure['score'][i,0]=0.
np.savez_compressed(rec/'hand_measurements.npz',**measure)
old_hands=motion['hand_joint_positions'].copy();wrist=missing_path(old_hands[:,0,0]);shape_a=camera_points(old_hands[lo,0],lo)-camera_points(old_hands[lo,0,0],lo);shape_b=camera_points(old_hands[hi,0],hi)-camera_points(old_hands[hi,0,0],hi)
for i in ids:
 t=(i-lo)/(hi-lo);shape=(1-t)*shape_a+t*shape_b;motion['hand_joint_positions'][i,0]=shape@camera[i,:3,:3].T+wrist[i]
motion['observed'][bad,0]=False;np.savez_compressed(rec/'scene_motion_physical.npz',**motion)
kin=setup_sharpa_kinematics('left',frequency=50.,max_iter=100,frame_tasks_converged_threshold=2e-6);reports=[]
for reference in list(dict.fromkeys([c['training_reference'],'track3_video','track3_coherent_variant','track3_terminal_supported_candidate'])):
 folder=r/f'human_motion_data/{reference}/processed'
 if not folder.exists():continue
 path=next(folder.rglob('motion.parquet'));row=pq.ParquetFile(path).read().to_pylist()[0];old={k:np.array(v) for k,v in row.items() if k.startswith(('object_body_','robot_right_','robot_left_'))};w=np.array(row['robot_left_wrist_position']);q=np.array(row['robot_left_wrist_wxyz']);f=np.array(row['robot_left_finger_joints']);frames=np.array(row['robot_left_frames']);w=missing_path(w);rotation=Slerp([0,1],Rotation.from_quat([q[lo],q[hi]],scalar_first=True));names=row['left_robot_frame_names']
 for i in ids:
  t=(i-lo)/(hi-lo);q[i]=rotation([t]).as_quat(scalar_first=True)[0];f[i]=(1-t)*f[lo]+t*f[hi];kin.configuration.update(np.r_[w[i],q[i,[1,2,3,0]],f[i]])
  for j,name in enumerate(names):
   M=kin.configuration.get_transform_frame_to_world(name);frames[i,j]=np.r_[M.translation,Rotation.from_matrix(M.rotation).as_quat(scalar_first=True)]
 for key,value in [('wrist_position',w),('wrist_wxyz',q),('finger_joints',f),('frames',frames)]:row[f'robot_left_{key}']=value.tolist()
 for suffix in ['positions','normals','part_ids']:
  key=f'mano_left_object_contact_{suffix}';value=np.array(row[key]);assert np.all(value[lo+1:hi]==0),key
 for key in old:
  if not key.startswith('robot_left_'):assert np.array_equal(old[key],np.array(row[key])),key
  else:assert np.array_equal(old[key][:lo+1],np.array(row[key])[:lo+1]) and np.array_equal(old[key][hi:],np.array(row[key])[hi:]),key
 pq.write_table(pa.Table.from_pylist([row]),path,compression='zstd');owner=r.stat();os.chown(path,owner.st_uid,owner.st_gid);reports.append({'reference':reference,'left_free_frames_changed':[lo+1,hi-1],'held_grasps_all_objects_and_entire_right_hand_preserved':True,'finger_posture_and_wrist_orientation_interpolated_only_when_not_visible':True})
report={'method':__doc__,'excluded_bystander_detection_frames':bad,'actual_wearer_missing_interval':[lo+1,hi-1],'own_hand_observation_anchors':[lo,hi],'source_hidden_motion_is_inferred':True,'outside_camera_wrist_y_px':980.,'camera_and_all_object_poses_unchanged':True,'reference_changes':reports};(r/'research/occluded-wearer-hand-repair.json').write_text(json.dumps(report,indent=2));print(report,flush=True)
