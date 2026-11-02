"""Fit initial support from an unassisted diagnostic before the first hand contact.
This edits only an unfrozen reconstructed reference; it never edits an executing policy state.
"""
import argparse,json,importlib.util
from pathlib import Path
import numpy as np,pyarrow as pa,pyarrow.parquet as pq,trimesh
from scipy.spatial.transform import Rotation,Slerp
from improvement_common import save,utc
p=argparse.ArgumentParser();p.add_argument('--run',type=Path,required=True);a=p.parse_args();r=a.run;c=json.loads((r/'episode-config.json').read_text());assert not (r/'coherent-input-checksums.json').exists();path=next((r/'human_motion_data/track3_collision_variant').rglob('motion.parquet'));row=pq.ParquetFile(path).read().to_pylist()[0];d=np.load(r/'evaluation/rebuild-passive-rest-states.npz');report=json.loads((r/'evaluation/rebuild-passive-rest-initial-rest.json').read_text());names=d['body_names'].tolist();T=c['frames'];P=np.array(row['object_body_position']);Q=np.array(row['object_body_wxyz']);changes=[]
spec=importlib.util.spec_from_file_location('fresh_geometry',r/'scripts/source_geometry.py');geo=importlib.util.module_from_spec(spec);spec.loader.exec_module(geo);m=np.load(r/'reconstruction/scene_motion_physical.npz');h=np.load(r/'reconstruction/hand_measurements.npz');C=np.linalg.inv(m['camera_to_world'][0])
def mat(p,q):M=np.eye(4);M[:3,:3]=Rotation.from_quat(q,scalar_first=True).as_matrix();M[:3,3]=p;return M
for item in report['objects']:
 if item['passed_1cm_0_2rad_passive_rest']:continue
 name=item['object'];b=c['objects'].index(name);start=min((g['start'] for g in c['grasps'] if g['object']==b),default=T);cutoff=item['control_frames_before_approach'];touch=item['first_hand_contact_control_frame']
 if cutoff<4 or (touch is not None and touch<cutoff):continue
 i=names.index('track3_'+name+'/object');settled=d['body_poses'][cutoff-1,i];before=mat(P[0,b],Q[0,b]);after=np.eye(4);after[:3,:3]=Rotation.from_quat(settled[3:]).as_matrix();after[:3,3]=settled[:3];delta=after@np.linalg.inv(before);translation=float(np.linalg.norm(after[:3,3]-before[:3,3]));angle=float((Rotation.from_matrix(before[:3,:3]).inv()*Rotation.from_matrix(after[:3,:3])).magnitude())
 if translation>.025 or angle>.25:
  # Approach the settled pose in a bounded step, then measure the remaining
  # drift again. Do not reject an otherwise useful repair just because the
  # full gravity-settled displacement slightly exceeds the edit bound.
  if translation>.075 or angle>.7:continue
  fraction=min(1.,.024/max(translation,1e-12),.24/max(angle,1e-12))
  after[:3,3]=before[:3,3]+fraction*(after[:3,3]-before[:3,3])
  after[:3,:3]=Slerp([0.,1.],Rotation.from_matrix([before[:3,:3],after[:3,:3]]))([fraction]).as_matrix()[0]
  delta=after@np.linalg.inv(before);translation=float(np.linalg.norm(after[:3,3]-before[:3,3]));angle=float((Rotation.from_matrix(before[:3,:3]).inv()*Rotation.from_matrix(after[:3,:3])).magnitude())
 import cv2
 vertices=trimesh.load(r/f'reconstruction/mesh_priors/{name}.ply',process=False).vertices;target=cv2.resize(cv2.imread(str(r/f'reconstruction/masks/{b+1}/000000.png'),0),(512,384),interpolation=cv2.INTER_NEAREST)>0;valid=~geo.human_occlusion(h,0)
 for other in range(len(c['objects'])):
  if other!=b:valid&=cv2.resize(cv2.imread(str(r/f'reconstruction/masks/{other+1}/000000.png'),0),(512,384),interpolation=cv2.INTER_NEAREST)==0
 score_before=geo.overlap(geo.silhouette(vertices,C@before),target,valid)['iou'];score_after=geo.overlap(geo.silhouette(vertices,C@after),target,valid)['iou']
 if score_after<score_before-.02:
  # Gravity fixes the support tilt; fit planar position/yaw back to the visible source mask.
  from scipy.optimize import differential_evolution
  settled=after.copy()
  def pose(x):
   M=settled.copy();M[:3,:3]=Rotation.from_euler('z',x[2]).as_matrix()@settled[:3,:3];M[:2,3]+=x[:2];return M
  def loss(x):return -geo.overlap(geo.silhouette(vertices,C@pose(x)),target,valid)['iou']
  fit=differential_evolution(loss,[(-.012,.012),(-.012,.012),(-.1,.1)],popsize=8,maxiter=18,polish=False,seed=20261022)
  if -fit.fun>score_after:after=pose(fit.x);score_after=-float(fit.fun)
  delta=after@np.linalg.inv(before);translation=float(np.linalg.norm(after[:3,3]-before[:3,3]));angle=float((Rotation.from_matrix(before[:3,:3]).inv()*Rotation.from_matrix(after[:3,:3])).magnitude())
  if translation>.025 or angle>.25:continue
 if score_after<score_before-.02:continue
 end=min(T-1,start+4)
 for frame in range(end+1):
  x=float(np.clip((frame-max(0,start-3))/max(1,end-max(0,start-3)),0,1));x=x*x*(3-2*x);rotation=Slerp([0,1],Rotation.from_matrix([delta[:3,:3],np.eye(3)]))([x]).as_matrix()[0];D=np.eye(4);D[:3,:3]=rotation;D[:3,3]=(1-x)*delta[:3,3];original=mat(P[frame,b],Q[frame,b]);changed=D@original;P[frame,b]=changed[:3,3];Q[frame,b]=Rotation.from_matrix(changed[:3,:3]).as_quat(scalar_first=True)
  for side in ['left','right']:
   if not any(g['object']==b and g['side']==side and g['start']<=frame<=g['end'] for g in c['grasps']):continue
   wrist=mat(row[f'robot_{side}_wrist_position'][frame],row[f'robot_{side}_wrist_wxyz'][frame]);wrist=D@wrist;row[f'robot_{side}_wrist_position'][frame]=wrist[:3,3].tolist();row[f'robot_{side}_wrist_wxyz'][frame]=Rotation.from_matrix(wrist[:3,:3]).as_quat(scalar_first=True).tolist();frames=np.array(row[f'robot_{side}_frames'][frame]);frames[:,:3]=frames[:,:3]@D[:3,:3].T+D[:3,3];frames[:,3:]=Rotation.from_matrix(D[:3,:3]@Rotation.from_quat(frames[:,3:],scalar_first=True).as_matrix()).as_quat(scalar_first=True);row[f'robot_{side}_frames'][frame]=frames.tolist()
  for side in ['left','right']:
   ids=np.array(row[f'mano_{side}_object_contact_part_ids'][frame]);active=ids==b+1
   for field in ['positions','normals']:
    values=np.array(row[f'mano_{side}_object_contact_{field}'][frame]);values[active]=values[active]@D[:3,:3].T+(D[:3,3] if field=='positions' else 0);row[f'mano_{side}_object_contact_{field}'][frame]=values.tolist()
 changes.append({'object':name,'translation_m':translation,'rotation_rad':angle,'source_iou_before':score_before,'source_iou_after':score_after,'unassisted_precontact_control_frames':cutoff,'initial_support_refinement_fades_by_source_frame':end})
row['object_body_position']=P.tolist();row['object_body_wxyz']=Q.tolist()
if changes:pq.write_table(pa.Table.from_pylist([row]),path,compression='zstd')
marker=r/'research/rebuilt-initial-support-refinement.json';history=json.loads(marker.read_text()).get('attempts',[]) if marker.exists() else [];history.append({'updated_utc':utc(),'changes':changes});save(marker,{'attempts':history,'actual_policy_states_or_forces_edited':False,'raw_source_reconstruction_unchanged':True});print('Bounded precontact support refinements',changes,flush=True)
