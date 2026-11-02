"""Test finer brush collision geometry and a raised, rack-clear grasp before learning."""
import os,json,copy,shutil,subprocess,traceback,xml.etree.ElementTree as ET
import numpy as np,pyarrow as pa,pyarrow.parquet as pq,trimesh
from scipy.spatial.transform import Rotation,Slerp
from improvement_common import BASE,BATCH,save,utc,digest,better
from batch_native import _docker,docker
from batch_review import review
from video_catalog import catalog
R=BASE/'runs/episode_000030_newton_consistent_brush_20261021';REF='track3_brush_grip_calibrated';A=BATCH/'improvements_20261021/activities/episode_000030-calibrated-rollout.json'
def status(state,phase,**extra):save(A,{'episode':30,'pid':os.getpid(),'process_script':__file__.split('/')[-1],'updated_utc':utc(),'status':state,'phase':phase,'description':'Evaluating one continuous policy rollout with the best contact-calibrated grip. The right hand follows the retargeted trajectory; the left uses the trained actor. No object guidance or failure resets.',**extra})
def main():
 status('running','Preparing a full-motion test of the best available contact-calibrated grasp; static acceptance was not achieved')
 target=R/'human_motion_data'/REF;assert not target.exists();shutil.copytree(R/'human_motion_data/track3_fine_collision_brush_candidate',target);path=next(target.rglob('motion.parquet'));row=pq.ParquetFile(path).read().to_pylist()[0]
 fit=json.loads((R/'evaluation/compliant-brush-20-feedback-braced-grasp-source89.json').read_text())['best_candidates']['mini_sweeper']
 save(R/'research/calibrated-brush-rollout-inputs.json',{'updated_utc':utc(),'reference':REF,'static_diagnostic_passed':fit['stable_hold_and_lift'],'reason_for_full_motion_check':'The near-feasible diagnostic grip is evaluated as a continuous policy attempt without changing any publication threshold. No additional learner is started.','physical_grip_fit':fit,'bristle_material_profile':{'stiffness':20.,'damping':1.,'tip_max_z':-.015},'robot_wrist_calibration':{'left':{'kp':25.,'kd':6.,'angular_kp':25.},'right':{'kp':500.,'kd':30.,'angular_kp':100.}}})
 status('running','Rebuilding a continuous reference with the independently tested raised grip')
 local=np.asarray(fit['wrist_in_object']);finger=np.asarray(fit['finger_joints']);P=np.asarray(row['object_body_position']);Q=np.asarray(row['object_body_wxyz']);w=np.asarray(row['robot_right_wrist_position']);q=np.asarray(row['robot_right_wrist_wxyz']);f=np.asarray(row['robot_right_finger_joints']);T=len(P)
 for i in range(26,254):
  O=np.eye(4);O[:3,:3]=Rotation.from_quat(Q[i,0],scalar_first=True).as_matrix();O[:3,3]=P[i,0];H=O@local;weight=min(float(np.clip((i-26)/8,0,1)),float(np.clip((253-i)/8,0,1)));weight=weight*weight*(3-2*weight);w[i]=(1-weight)*w[i]+weight*H[:3,3]
  if i<34:w[i,2]+=.04*np.sin(np.pi*(i-26)/8)
  q[i]=Slerp([0,1],Rotation.from_matrix([Rotation.from_quat(q[i],scalar_first=True).as_matrix(),H[:3,:3]]))([weight]).as_quat(scalar_first=True)[0];closure=min(float(np.clip((i-31)/3,0,1)),float(np.clip((250-i)/5,0,1)));closure=closure*closure*(3-2*closure);f[i]=(1-closure)*f[i]+closure*finger
 row['robot_right_wrist_position']=w.tolist();row['robot_right_wrist_wxyz']=q.tolist();row['robot_right_finger_joints']=f.tolist();pq.write_table(pa.Table.from_pylist([row]),path,compression='zstd')
 image=_docker(['inspect','v2d-batch-episode32-retarget','--format','{{.Image}}']).stdout.strip()
 _docker(['run','--rm','--name','v2d-fine-brush-fk-cpu','--ipc','host','-e','CUDA_VISIBLE_DEVICES=','-e','V2D_TRANSITION_REFERENCE='+REF,'-e','V2D_REBUILD_ONLY=1','-e','V2D_CONTACT_APPLY_OUTPUT=research/fine-brush-fk.json','-v',f'{R}:/episode','-v',f'{BASE}/video_to_data/robotic_grounding/source/robotic_grounding/robotic_grounding/assets:/v2d/assets:ro','--entrypoint','/isaac-sim/python.sh',image,'-u','/episode/scripts/apply_collision_transitions.py'],R/'logs/fine-brush-fk.log')
 status('running','Independent whole-source32-world rollout; no object assistance')
 profile={'left':0.,'right':.5,'maximum_correction_m':.015,'right_wrist_residual_gain':0.,'right_finger_residual_gain':0.};profile['left']=2.
 candidate=review(30,45001728,reference_override=REF,root_override=R,container_override='v2d-consistent-episode30-newton',substeps_override=20,recording_world_count=32,tracking_feedback=profile,bristle_compliance={'stiffness':20.,'damping':1.,'tip_max_z':-.015},wrist_calibration={'left':{'kp':25.,'kd':6.,'angular_kp':25.},'right':{'kp':500.,'kd':30.,'angular_kp':100.}});previous=catalog(30,BASE/'runs/episode_000030_newton_batch_20261015',True)['best']['selected'];eligible=better(candidate,previous);candidate['passes_current_accepted_policy_gate']=eligible;save(R/'research/calibrated-brush-policy-review.json',candidate);status('completed','Finer collision and raised-grip policy review complete',eligible_for_independent_publication=eligible,source_mean_object_iou=candidate['source_mean_object_iou'],completion_rate=candidate['completion_rate_32_worlds']);print('Fine brush policy',candidate['source_mean_object_iou'],candidate['completion_rate_32_worlds'],'better',eligible,flush=True)
if __name__=='__main__':
 try:main()
 except Exception:status('failed','Finer collision experiment failed; retained video unchanged',error=traceback.format_exc());raise
