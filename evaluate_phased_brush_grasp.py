"""Reconstruct a pickup-to-brushing grip transition and test one continuous rollout."""
import os,json,shutil,subprocess,traceback
import numpy as np,pyarrow as pa,pyarrow.parquet as pq
from scipy.spatial.transform import Rotation,Slerp
from improvement_common import BASE,BATCH,save,utc,better
from batch_native import _docker
from batch_review import review
from video_catalog import catalog
R=BASE/'runs/episode_000030_newton_consistent_brush_20261021';REF='track3_brush_phased_grasp';A=BATCH/'improvements_20261021/activities/episode_000030-phased-grip-rollout.json'
def status(state,phase,**extra):save(A,{'episode':30,'pid':os.getpid(),'process_script':__file__.split('/')[-1],'updated_utc':utc(),'status':state,'phase':phase,'description':'Testing the independently checked pickup grip followed by a higher brushing grip. The robot hand changes grip smoothly; object poses are never reset or guided.',**extra})
def smooth(x):x=float(np.clip(x,0,1));return x*x*(3-2*x)
def main():
 status('running','Rebuilding the video-timed pickup-to-brushing grip transition')
 early=json.loads((R/'evaluation/compliant-brush-20-pickup-grasp-source34.json').read_text())['best_candidates']['mini_sweeper'];assert early['stable_hold_and_lift']
 late=json.loads((R/'research/brush-local-stable-late-grip.json').read_text())['fit']
 target=R/'human_motion_data'/REF;assert not target.exists();shutil.copytree(R/'human_motion_data/track3_fine_collision_brush_candidate',target);path=next(target.rglob('motion.parquet'));row=pq.ParquetFile(path).read().to_pylist()[0]
 P=np.asarray(row['object_body_position']);Q=np.asarray(row['object_body_wxyz']);w=np.asarray(row['robot_right_wrist_position']);q=np.asarray(row['robot_right_wrist_wxyz']);f=np.asarray(row['robot_right_finger_joints']);E=np.asarray(early['wrist_in_object']);L=np.asarray(late['wrist_in_object']);blend_rotation=Slerp([0,1],Rotation.from_matrix([E[:3,:3],L[:3,:3]]))
 for i in range(26,254):
  blend=smooth((i-50)/15);local=np.eye(4);local[:3,:3]=blend_rotation([blend]).as_matrix()[0];local[:3,3]=(1-blend)*E[:3,3]+blend*L[:3,3];O=np.eye(4);O[:3,:3]=Rotation.from_quat(Q[i,0],scalar_first=True).as_matrix();O[:3,3]=P[i,0];H=O@local
  weight=smooth((i-26)/8)*smooth((253-i)/8);w[i]=(1-weight)*w[i]+weight*H[:3,3]
  if i<34:w[i,2]+=.04*np.sin(np.pi*(i-26)/8)
  q[i]=Slerp([0,1],Rotation.from_matrix([Rotation.from_quat(q[i],scalar_first=True).as_matrix(),H[:3,:3]]))([weight]).as_quat(scalar_first=True)[0]
  fingers=(1-blend)*np.asarray(early['finger_joints'])+blend*np.asarray(late['finger_joints']);closure=smooth((i-31)/3)*smooth((250-i)/5);f[i]=(1-closure)*f[i]+closure*fingers
 row['robot_right_wrist_position']=w.tolist();row['robot_right_wrist_wxyz']=q.tolist();row['robot_right_finger_joints']=f.tolist();pq.write_table(pa.Table.from_pylist([row]),path,compression='zstd')
 save(R/'research/phased-brush-grip-reconstruction.json',{'updated_utc':utc(),'reference':REF,'pickup_fit':early,'brushing_fit':late,'pickup_grip_passed_strict_two_object_test':True,'brushing_grip_passed_brush_only_test':True,'brushing_grip_rack_drift_still_requires_full_policy_check':True,'grip_transition_source_interval':[50,65],'source_object_poses_camera_and_clock_unchanged':True,'no_policy_training_added':True,'control':'Learned left-hand actor; reference-relative right-hand trajectory with bounded live object tracking feedback.'})
 image=_docker(['inspect','v2d-batch-episode32-retarget','--format','{{.Image}}']).stdout.strip();_docker(['run','--rm','--name','v2d-phased-brush-fk-cpu','--ipc','host','-e','CUDA_VISIBLE_DEVICES=','-e','V2D_TRANSITION_REFERENCE='+REF,'-e','V2D_REBUILD_ONLY=1','-e','V2D_CONTACT_APPLY_OUTPUT=research/phased-brush-fk.json','-v',f'{R}:/episode','-v',f'{BASE}/video_to_data/robotic_grounding/source/robotic_grounding/robotic_grounding/assets:/v2d/assets:ro','--entrypoint','/isaac-sim/python.sh',image,'-u','/episode/scripts/apply_collision_transitions.py'],R/'logs/phased-brush-fk.log')
 status('running','Independent whole-source32-world hybrid policy attempt; no object guidance or resets')
 row=review(30,45001728,reference_override=REF,root_override=R,container_override='v2d-consistent-episode30-newton',substeps_override=20,recording_world_count=32,tracking_feedback={'left':2.,'right':.5,'maximum_correction_m':.03,'right_wrist_residual_gain':0.,'right_finger_residual_gain':0.},bristle_compliance={'stiffness':20.,'damping':1.,'tip_max_z':-.015},wrist_calibration={'left':{'kp':25.,'kd':6.,'angular_kp':25.},'right':{'kp':500.,'kd':30.,'angular_kp':100.}})
 accepted=catalog(30,BASE/'runs/episode_000030_newton_batch_20261015',True)['best']['selected'];eligible=better(row,accepted);row['passes_current_accepted_policy_gate']=eligible;save(R/'research/phased-brush-policy-review.json',row);status('completed','Phased-grip continuous rollout reviewed',eligible_for_independent_publication=eligible,source_mean_object_iou=row['source_mean_object_iou'],completion_rate=row['completion_rate_32_worlds']);print('Phased brush meanIoU',row['source_mean_object_iou'],'nativeSR',row['completion_rate_32_worlds'],'better',eligible,flush=True)
if __name__=='__main__':
 try:main()
 except Exception:status('failed','Phased brush evaluation failed; previous video retained',error=traceback.format_exc());raise
