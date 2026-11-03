"""Choose an independent coherent posture for every source-observed grasp interval."""
import json,shutil
from pathlib import Path
import numpy as np,pyarrow as pa,pyarrow.parquet as pq,trimesh
from scipy.spatial import cKDTree
from scipy.spatial.transform import Rotation,Slerp
from robotic_grounding.retarget.retarget_utils import setup_sharpa_kinematics
R=Path('/episode');cfg=json.loads((R/'episode-config.json').read_text());seq=cfg['sequence'];rec=R/'reconstruction';p=R/f'human_motion_data/track3_video/processed/sequence_id={seq}/robot_name=sharpa_wave/motion.parquet';c=pq.read_table(p).to_pylist()[0];m=np.load(rec/'scene_motion_physical.npz');O=m['object_to_world'];H=m['hand_joint_positions'];T=len(H);report=[];meshes=[trimesh.load(rec/f'mesh_priors/{n}.ply',process=False) for n in cfg['objects']];trees=[cKDTree(v.vertices) for v in meshes]
for s,side in enumerate(['left','right']):
 frames=np.array(c[f'robot_{side}_frames']);names=c[f'{side}_robot_frame_names'];w=np.array(c[f'robot_{side}_wrist_position']);Q=np.array(c[f'robot_{side}_wrist_wxyz']);f=np.array(c[f'robot_{side}_finger_joints']);kin=setup_sharpa_kinematics(side,frequency=50.,max_iter=100,frame_tasks_converged_threshold=2e-6)
 indices=[names.index(f'{side}_hand_C_MC')]
 for finger in ['thumb','index','middle','ring','pinky']:
  segments=['CMC_VL','MCP_VL','DP'] if finger=='thumb' else ['MCP_VL','MP','DP'];indices.extend([names.index(f'{side}_{finger}_{x}') for x in segments]+[names.index(f'{side}_{finger}_tip_site')])
 for g in [g for g in cfg['grasps'] if g['side']==side]:
  start,end,b=g['start'],g['end'],g['object'];lo=max(0,start-5);hi=min(T-1,end+5)
  local=np.einsum('nij,nkj->nki',O[start:end+1,b,:3,:3].transpose(0,2,1),H[start:end+1,s]-O[start:end+1,b,None,:3,3]);target=np.median(local,axis=0);scores=[]
  for t in range(start+min(10,(end-start)//4),end-min(10,(end-start)//4)+1):
   P=(frames[t,indices,:3]-O[t,b,:3,3])@O[t,b,:3,:3];scores.append((float(np.mean(np.linalg.norm(P-target,axis=1))),t))
  score,t=min(scores);base=np.eye(4);base[:3,:3]=Rotation.from_quat(Q[t],scalar_first=True).as_matrix();base[:3,3]=w[t];localbase=np.linalg.inv(O[t,b])@base;finger=f[t].copy()
  for i in range(lo,hi+1):
   blend=float(np.clip(min((i-lo)/max(1,start-lo),(hi-i)/max(1,hi-end)),0,1))
   if start<=i<=end:blend=1.
   new=O[i,b]@localbase;w[i]=(1-blend)*w[i]+blend*new[:3,3];Q[i]=Slerp([0,1],Rotation.from_matrix([Rotation.from_quat(Q[i],scalar_first=True).as_matrix(),new[:3,:3]]))([blend]).as_quat(scalar_first=True)[0];f[i]=(1-blend)*f[i]+blend*finger
  report.append({'side':side,'object':cfg['objects'][b],'template_source_frame':t,'mean_anatomical_landmark_error_m':score,'held_interval':[start,end],'blend_interval':[lo,hi]})
 # Recompute all FK and all contact labels from the final grasp targets.
 for i in range(T):
  kin.configuration.update(np.r_[w[i],Q[i,[1,2,3,0]],f[i]])
  for j,n in enumerate(names):
   tr=kin.configuration.get_transform_frame_to_world(n);frames[i,j]=np.r_[tr.translation,Rotation.from_matrix(tr.rotation).as_quat(scalar_first=True)]
 cp=np.zeros((T,5,3));cn=np.zeros_like(cp);pid=np.zeros((T,5),int);tips=[names.index(f'{side}_{finger}_tip_site') for finger in ['thumb','index','middle','ring','pinky']]
 for i in range(T):
  active=[g['object'] for g in cfg['grasps'] if g['side']==side and max(0,g['start']-3)<=i<=min(T-1,g['end']+3)]
  for j,tip in enumerate(tips):
   found=[]
   for b in active:
    M=O[i,b];dist,idx=trees[b].query((frames[i,tip,:3]-M[:3,3])@M[:3,:3]);found.append((dist,b,idx))
   if not found:continue
   dist,b,idx=min(found)
   if dist>.025:continue
   M=O[i,b];n=meshes[b].vertex_normals[idx]@M[:3,:3].T;length=np.linalg.norm(n)
   if not np.isfinite(length) or length<1e-8:continue
   cp[i,j]=meshes[b].vertices[idx]@M[:3,:3].T+M[:3,3];cn[i,j]=n/length;pid[i,j]=b+1
 for key,value in [('wrist_position',w),('wrist_wxyz',Q),('finger_joints',f),('frames',frames)]:c[f'robot_{side}_{key}']=value.tolist()
 for key,value in [('positions',cp),('normals',cn),('part_ids',pid)]:c[f'mano_{side}_object_contact_{key}']=value.tolist()
out=R/f'human_motion_data/track3_coherent_variant/processed/sequence_id={seq}/robot_name=sharpa_wave';out.mkdir(parents=True,exist_ok=True);pq.write_table(pa.Table.from_pylist([c]),out/'motion.parquet',compression='zstd');stage=out.parent.parent.parent/'reconstructed_stage';stage.mkdir(exist_ok=True);shutil.copy(p.parent.parent.parent.parent/f'reconstructed_stage/{seq}_support.usda',stage);(R/'research/coherent-grasp-variant.json').write_text(json.dumps(report,indent=2));print(json.dumps(report,indent=2))

# Give the workspace owner control of generated references and directories.
import os
owner=R.stat()
for parent,dirs,files in os.walk(R/"human_motion_data"):
 os.chown(parent,owner.st_uid,owner.st_gid)
 for item in files:os.chown(os.path.join(parent,item),owner.st_uid,owner.st_gid)
