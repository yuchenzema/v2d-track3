"""Apply collision-optimized grasps while preserving each video-derived object trajectory."""
import json,shutil,os
from pathlib import Path
import numpy as np,pyarrow as pa,pyarrow.parquet as pq,trimesh
from scipy.spatial import cKDTree
from scipy.spatial.transform import Rotation,Slerp
from robotic_grounding.retarget.retarget_utils import setup_sharpa_kinematics
from rigid_pose_collision_audit import unfitted_blend_frames
R=Path('/episode');cfg=json.loads((R/'episode-config.json').read_text());seq=cfg['sequence'];rec=R/'reconstruction';reference=os.environ.get('V2D_TRANSITION_REFERENCE','track3_collision_variant');p=R/f'human_motion_data/{reference}/processed/sequence_id={seq}/robot_name=sharpa_wave/motion.parquet';c=pq.ParquetFile(p).read().to_pylist()[0];m=np.load(rec/'scene_motion_physical.npz');O=m['object_to_world'];H=m['hand_joint_positions'];T=len(H);report=[];meshes=[trimesh.load(rec/f'mesh_priors/{n}.ply',process=False) for n in cfg['objects']];trees=[cKDTree(v.vertices) for v in meshes]
O=np.tile(np.eye(4),(T,len(cfg['objects']),1,1));O[:,:,:3,3]=np.array(c['object_body_position']);O[:,:,:3,:3]=Rotation.from_quat(np.array(c['object_body_wxyz']).reshape(-1,4),scalar_first=True).as_matrix().reshape(T,len(cfg['objects']),3,3)
if os.environ.get('V2D_CONTACT_GRASPS_OVERRIDE'):
 cfg['grasps']=json.loads(os.environ['V2D_CONTACT_GRASPS_OVERRIDE'])
 for g in cfg['grasps']:assert g['side'] in ['left','right'] and 0<=g['object']<len(cfg['objects']) and 0<=g['start']<=g['end']<T
for s,side in enumerate(['left','right']):
 frames=np.array(c[f'robot_{side}_frames']);names=c[f'{side}_robot_frame_names'];w=np.array(c[f'robot_{side}_wrist_position']);Q=np.array(c[f'robot_{side}_wrist_wxyz']);f=np.array(c[f'robot_{side}_finger_joints']);kin=setup_sharpa_kinematics(side,frequency=50.,max_iter=100,frame_tasks_converged_threshold=2e-6)
 indices=[names.index(f'{side}_hand_C_MC')]
 for finger in ['thumb','index','middle','ring','pinky']:
  segments=['CMC_VL','MCP_VL','DP'] if finger=='thumb' else ['MCP_VL','MP','DP'];indices.extend([names.index(f'{side}_{finger}_{x}') for x in segments]+[names.index(f'{side}_{finger}_tip_site')])
 fits=[] if os.environ.get('V2D_REBUILD_ONLY')=='1' else json.loads((R/os.environ.get('V2D_CONTACT_FIT_OUTPUT','research/collision-transition-poses.json')).read_text())['grasps']
 for fit in [g for g in fits if g['side']==side]:
  start,end=fit['held_interval'];b=fit['object_index'];lo=start;hi=end;localbase=np.array(fit['wrist_in_object']);finger=np.array(fit['finger_joints'])
  for i in range(lo,hi+1):
   blend=float(np.clip(min((i-lo)/max(1,start-lo),(hi-i)/max(1,hi-end)),0,1))
   if start<=i<=end:blend=1.
   new=O[i,b]@localbase;w[i]=(1-blend)*w[i]+blend*new[:3,3];Q[i]=Slerp([0,1],Rotation.from_matrix([Rotation.from_quat(Q[i],scalar_first=True).as_matrix(),new[:3,:3]]))([blend]).as_quat(scalar_first=True)[0];f[i]=(1-blend)*f[i]+blend*finger
  report.append(fit)
 if os.environ.get('V2D_BLEND_ADJACENT_FITS')=='1':
  ordered=sorted([g for g in fits if g['side']==side],key=lambda g:g['held_interval'][0])
  for first,second in zip(ordered[:-1],ordered[1:]):
   if first['object_index']!=second['object_index'] or first['held_interval'][1]+1!=second['held_interval'][0]:continue
   lo,hi,blend_frames=unfitted_blend_frames(first,second,ordered,T);A=np.array(first['wrist_in_object']);B=np.array(second['wrist_in_object']);b=first['object_index']
   for i in blend_frames:
    blend=(i-lo)/max(1,hi-lo);local=np.eye(4);local[:3,3]=(1-blend)*A[:3,3]+blend*B[:3,3];local[:3,:3]=Slerp([0,1],Rotation.from_matrix([A[:3,:3],B[:3,:3]]))([blend])[0].as_matrix();world=O[i,b]@local;w[i]=world[:3,3];Q[i]=Rotation.from_matrix(world[:3,:3]).as_quat(scalar_first=True);f[i]=(1-blend)*np.array(first['finger_joints'])+blend*np.array(second['finger_joints'])
 # Recompute all FK and all contact labels from the final grasp targets.
 for i in range(T):
  kin.configuration.update(np.r_[w[i],Q[i,[1,2,3,0]],f[i]])
  for j,n in enumerate(names):
   tr=kin.configuration.get_transform_frame_to_world(n);frames[i,j]=np.r_[tr.translation,Rotation.from_matrix(tr.rotation).as_quat(scalar_first=True)]
 cp=np.zeros((T,5,3));cn=np.zeros_like(cp);pid=np.zeros((T,5),int);tips=[names.index(f'{side}_{finger}_tip_site') for finger in ['thumb','index','middle','ring','pinky']]
 for i in range(T):
  active=[g['object'] for g in cfg['grasps'] if g['side']==side and max(0,g['start']-3)<=i<=g['end']]
  for j,tip in enumerate(tips):
   found=[]
   for b in active:
    M=O[i,b];dist,idx=trees[b].query((frames[i,tip,:3]-M[:3,3])@M[:3,:3]);found.append((dist,b,idx))
   if not found:continue
   dist,b,idx=min(found)
   if dist>.025:continue
   M=O[i,b];cp[i,j]=meshes[b].vertices[idx]@M[:3,:3].T+M[:3,3];n=meshes[b].vertex_normals[idx]@M[:3,:3].T;cn[i,j]=n/(np.linalg.norm(n)+1e-12);pid[i,j]=b+1
 for key,value in [('wrist_position',w),('wrist_wxyz',Q),('finger_joints',f),('frames',frames)]:c[f'robot_{side}_{key}']=value.tolist()
 for key,value in [('positions',cp),('normals',cn),('part_ids',pid)]:c[f'mano_{side}_object_contact_{key}']=value.tolist()
out=R/f'human_motion_data/{reference}/processed/sequence_id={seq}/robot_name=sharpa_wave';out.mkdir(parents=True,exist_ok=True);pq.write_table(pa.Table.from_pylist([c]),out/'motion.parquet',compression='zstd');stage=out.parent.parent.parent/'reconstructed_stage';stage.mkdir(exist_ok=True);(R/os.environ.get('V2D_CONTACT_APPLY_OUTPUT','research/collision-transition-applied.json')).write_text(json.dumps(report,indent=2));print(json.dumps(report,indent=2))

# Give the workspace owner control of generated references and directories.
import os
owner=R.stat()
for parent,dirs,files in os.walk(R/"human_motion_data"):
 os.chown(parent,owner.st_uid,owner.st_gid)
 for item in files:os.chown(os.path.join(parent,item),owner.st_uid,owner.st_gid)
