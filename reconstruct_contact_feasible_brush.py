"""Fit a collision-free held brush pose to the source image, then check full motion."""
import os,json,shutil,subprocess,traceback
import numpy as np,pyarrow as pa,pyarrow.parquet as pq
from scipy.spatial.transform import Rotation
from improvement_common import BASE,BATCH,save,utc,better
from batch_native import _docker,docker
from batch_review import review
from video_catalog import catalog
R=BASE/'runs/episode_000030_newton_consistent_brush_20261021';REF='track3_contact_feasible_brush_candidate'
A=BATCH/'improvements_20261021/activities/episode_000030-contact-reconstruction.json'
def status(state,phase,**extra):save(A,{'episode':30,'pid':os.getpid(),'process_script':__file__.split('/')[-1],'updated_utc':utc(),'status':state,'phase':phase,'description':'Refining brush depth and orientation against the source image and actual rack/table collisions, then checking the whole rollout.',**extra})
def main():
 status('running','Native collision-aware image fit at the representative long-hold pose')
 target=R/'human_motion_data'/REF;assert not target.exists(),'Do not overwrite a previous reconstruction experiment'
 shutil.copytree(R/'human_motion_data/track3_consistent_brush_pickup_candidate',target);path=next(target.rglob('motion.parquet'));original=pq.ParquetFile(path).read().to_pylist()[0]
 shutil.copy2(BASE/'refine_native_object_poses.py',R/'scripts/refine_contact_feasible_brush.py')
 _docker(['exec','v2d-consistent-episode30-newton','env','CUDA_VISIBLE_DEVICES=','JAX_PLATFORMS=cpu','V2D_OBJECT_POSE_REFERENCE='+REF,'V2D_OBJECT_POSE_MOBILE_OBJECTS=[0]','V2D_PRESERVE_STATIC_POSES=1','V2D_OBJECT_POSE_START_FRAME=89','V2D_OBJECT_POSE_END_FRAME=89','V2D_OBJECT_POSE_COMPACT_CONTACTS=1','V2D_OBJECT_POSE_CONTACT_SIGMA=.0001','V2D_OBJECT_POSE_MAX_TRANSLATION=.05','V2D_OBJECT_POSE_MAX_ROTATION=.6','V2D_OBJECT_POSE_UPWARD_SEEDS=1','V2D_OBJECT_POSE_OUTPUT=research/contact-feasible-brush-anchor-fit.json','python','-u','/episode/scripts/run_geometry_cpu.py','/episode/scripts/refine_contact_feasible_brush.py'],R/'logs/contact-feasible-brush-anchor-fit.log')
 fitted=pq.ParquetFile(path).read().to_pylist()[0];p0=np.asarray(original['object_body_position']);q0=np.asarray(original['object_body_wxyz']);p=np.asarray(fitted['object_body_position']);q=np.asarray(fitted['object_body_wxyz']);T=len(p)
 rot0=Rotation.from_quat(q0[89,0],scalar_first=True);rot=Rotation.from_quat(q[89,0],scalar_first=True);delta_rot=rot0.inv()*rot
 import trimesh
 centre=np.asarray(trimesh.load(R/'reconstruction/mesh_priors/mini_sweeper.ply',process=False).vertices).mean(0)
 displacement=rot.apply(centre)+p[89,0]-(rot0.apply(centre)+p0[89,0]);row=original
 P=p0.copy();Q=q0.copy();D=np.tile(np.eye(3),(T,1,1));shift=np.zeros((T,3))
 for i in range(T):
  weight=min(float(np.clip((i-50)/15,0,1)),float(np.clip((245-i)/10,0,1)));weight=weight*weight*(3-2*weight)
  old=Rotation.from_quat(q0[i,0],scalar_first=True);new=old*Rotation.from_rotvec(weight*delta_rot.as_rotvec())
  P[i,0]=old.apply(centre)+p0[i,0]+weight*displacement-new.apply(centre);Q[i,0]=new.as_quat(scalar_first=True);D[i]=new.as_matrix()@old.as_matrix().T;shift[i]=P[i,0]-D[i]@p0[i,0]
 row['object_body_position']=P.tolist();row['object_body_wxyz']=Q.tolist()
 side='right';w=np.asarray(row[f'robot_{side}_wrist_position']);wq=np.asarray(row[f'robot_{side}_wrist_wxyz']);frames=np.asarray(row[f'robot_{side}_frames'])
 for i in range(T):
  w[i]=D[i]@w[i]+shift[i];wq[i]=(Rotation.from_matrix(D[i])*Rotation.from_quat(wq[i],scalar_first=True)).as_quat(scalar_first=True);frames[i,:,:3]=frames[i,:,:3]@D[i].T+shift[i];frames[i,:,3:]=(Rotation.from_matrix(D[i])*Rotation.from_quat(frames[i,:,3:],scalar_first=True)).as_quat(scalar_first=True)
 row[f'robot_{side}_wrist_position']=w.tolist();row[f'robot_{side}_wrist_wxyz']=wq.tolist();row[f'robot_{side}_frames']=frames.tolist()
 for side in ['left','right']:
  cp=np.asarray(row[f'mano_{side}_object_contact_positions']);cn=np.asarray(row[f'mano_{side}_object_contact_normals']);ids=np.asarray(row[f'mano_{side}_object_contact_part_ids'])
  for i in range(T):
   hit=ids[i]==1;cp[i,hit]=cp[i,hit]@D[i].T+shift[i];cn[i,hit]=cn[i,hit]@D[i].T
  row[f'mano_{side}_object_contact_positions']=cp.tolist();row[f'mano_{side}_object_contact_normals']=cn.tolist()
 pq.write_table(pa.Table.from_pylist([row]),path,compression='zstd')
 save(R/'research/contact-feasible-brush-reconstruction.json',{'updated_utc':utc(),'reference':REF,'anchor_source_frame':89,'source_frame_count':T,'held_centroid_correction_m':displacement.tolist(),'held_local_rotation_correction_rad':delta_rot.as_rotvec().tolist(),'initial_stationary_support_unchanged':True,'rack_camera_and_source_clock_unchanged':True,'existing_training_reference_unchanged':True,'diagnostic_only':True})
 status('running','Checking all263 source images and native reference collisions')
 with (R/'logs/contact-feasible-brush-source-fidelity.log').open('w') as log:subprocess.run([str(R/'.venv-video/bin/python'),str(BASE/'assess_physical_reference.py'),'--run',str(R),'--reference',REF,'--output','contact-feasible-brush-source-fidelity','--no-update-selection'],stdout=log,stderr=subprocess.STDOUT,check=True)
 _docker(['exec','v2d-consistent-episode30-newton','env','CUDA_VISIBLE_DEVICES=','JAX_PLATFORMS=cpu','V2D_AUDIT_REFERENCE='+REF,'python','-u','/episode/scripts/run_geometry_cpu.py','/episode/scripts/native_reference_audit.py'],R/'logs/contact-feasible-brush-collision-audit.log')
 status('running','Independent native32-world policy attempt on the corrected reference')
 profile={'left':0.,'right':.5,'maximum_correction_m':.015,'right_wrist_residual_gain':.25,'right_finger_residual_gain':.5}
 candidate=review(30,45001728,reference_override=REF,root_override=R,container_override='v2d-consistent-episode30-newton',substeps_override=20,recording_world_count=32,tracking_feedback=profile)
 previous=catalog(30,BASE/'runs/episode_000030_newton_batch_20261015',True)['best']['selected'];eligible=better(candidate,previous);candidate['passes_current_accepted_policy_gate']=eligible;save(R/'research/contact-feasible-brush-policy-review.json',candidate)
 status('completed','Contact-aware reconstruction and full-motion policy review complete',eligible_for_independent_publication=eligible,source_mean_object_iou=candidate['source_mean_object_iou'],completion_rate=candidate['completion_rate_32_worlds']);print('Brush corrected reconstruction review',candidate['source_mean_object_iou'],candidate['completion_rate_32_worlds'],'better',eligible,flush=True)
if __name__=='__main__':
 try:main()
 except Exception:status('failed','Contact-aware reconstruction failed; previous accepted video retained',error=traceback.format_exc());raise
