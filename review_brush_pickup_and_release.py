"""Delay closure until the pickup pose, and clear both hands after stable placement."""
import os,json,shutil,traceback
import numpy as np,pyarrow as pa,pyarrow.parquet as pq
from scipy.spatial.transform import Rotation,Slerp
from improvement_common import BASE,BATCH,save,utc,better
from batch_review import review
from video_catalog import catalog
from object_motion_quality import attach
R=BASE/'runs/episode_000030_newton_consistent_brush_20261021';REF='track3_brush_safe_release';A=BATCH/'improvements_20261021/activities/episode_000030-safe-release-rollout.json'
def status(state,phase,**extra):save(A,{'episode':30,'pid':os.getpid(),'process_script':__file__.split('/')[-1],'updated_utc':utc(),'status':state,'phase':phase,'description':'Testing delayed finger closure, upright placement in the tested rack support, and upward hand withdrawal. Original rigid object models and45N limits are retained.',**extra})
def smooth(x):x=float(np.clip(x,0,1));return x*x*(3-2*x)
def matrix(p,q):M=np.eye(4);M[:3,3]=p;M[:3,:3]=Rotation.from_quat(q,scalar_first=True).as_matrix();return M
def main():
 status('running','Reconstructing collision-aware pickup, placement and hand withdrawal')
 target=R/'human_motion_data'/REF;assert not target.exists();shutil.copytree(R/'human_motion_data/track3_brush_phased_coarse',target);path=next(target.rglob('motion.parquet'));row=pq.ParquetFile(path).read().to_pylist()[0];base=pq.ParquetFile(next((R/'human_motion_data/track3_consistent_brush_pickup_candidate').rglob('motion.parquet'))).read().to_pylist()[0]
 P=np.asarray(row['object_body_position']);Q=np.asarray(row['object_body_wxyz']);T=len(P);local=np.linalg.inv(matrix(P[0,1],Q[0,1]))@matrix(P[0,0],Q[0,0]);goal=matrix(P[235,1],Q[235,1])@local;alternative=goal[:3,:3]@Rotation.from_rotvec([0,0,np.pi]).as_matrix();held=Rotation.from_quat(Q[235,0],scalar_first=True)
 alternate=(held.inv()*Rotation.from_matrix(alternative)).magnitude()<(held.inv()*Rotation.from_matrix(goal[:3,:3])).magnitude()
 if alternate:local[:3,:3]=local[:3,:3]@Rotation.from_rotvec([0,0,np.pi]).as_matrix()
 for i in range(235,T):
  goal=matrix(P[i,1],Q[i,1])@local;blend=smooth((i-235)/10);P[i,0]=(1-blend)*P[i,0]+blend*goal[:3,3];Q[i,0]=Slerp([0,1],Rotation.from_matrix([Rotation.from_quat(Q[i,0],scalar_first=True).as_matrix(),goal[:3,:3]]))([blend]).as_quat(scalar_first=True)[0]
 row['object_body_position']=P.tolist();row['object_body_wxyz']=Q.tolist();early=np.linalg.inv(matrix(base['object_body_position'][34][0],base['object_body_wxyz'][34][0]))@matrix(base['robot_right_wrist_position'][34],base['robot_right_wrist_wxyz'][34]);late=np.asarray(json.loads((R/'research/brush-local-stable-late-grip.json').read_text())['fit']['wrist_in_object']);finger_early=np.asarray(base['robot_right_finger_joints'][34]);finger_late=np.asarray(json.loads((R/'research/brush-local-stable-late-grip.json').read_text())['fit']['finger_joints']);rot=Slerp([0,1],Rotation.from_matrix([early[:3,:3],late[:3,:3]]))
 w=np.asarray(base['robot_right_wrist_position']);q=np.asarray(base['robot_right_wrist_wxyz']);f=np.asarray(base['robot_right_finger_joints']);opened=f[0].copy()
 for i in range(26,246):
  blend=smooth((i-50)/15);grip=np.eye(4);grip[:3,:3]=rot([blend]).as_matrix()[0];grip[:3,3]=(1-blend)*early[:3,3]+blend*late[:3,3];H=matrix(P[i,0],Q[i,0])@grip;approach=smooth((i-26)/8);w[i]=(1-approach)*w[i]+approach*H[:3,3]
  if i<34:w[i,2]+=.05*np.sin(np.pi*(i-26)/8)
  q[i]=Slerp([0,1],Rotation.from_matrix([Rotation.from_quat(q[i],scalar_first=True).as_matrix(),H[:3,:3]]))([approach]).as_quat(scalar_first=True)[0];close=smooth((i-34)/4);f[i]=(1-close)*opened+close*((1-blend)*finger_early+blend*finger_late)
 row['robot_right_wrist_position']=w.tolist();row['robot_right_wrist_wxyz']=q.tolist();row['robot_right_finger_joints']=f.tolist()
 for side,hold_frame,open_start,open_end,lift_start,lift_end in [('right',245,246,249,250,255),('left',244,247,250,251,257)]:
  w=np.asarray(row[f'robot_{side}_wrist_position']);q=np.asarray(row[f'robot_{side}_wrist_wxyz']);f=np.asarray(row[f'robot_{side}_finger_joints']);held_pos=w[hold_frame].copy();held_q=q[hold_frame].copy();held_f=f[hold_frame].copy();opened=np.asarray(base[f'robot_{side}_finger_joints'][0]);free=np.asarray(base[f'robot_{side}_wrist_position'][-1]).copy();delta=free[:2]-P[-1,1,:2];delta/=max(np.linalg.norm(delta),1e-9);free[:2]=P[-1,1,:2]+delta*.55;free[2]=held_pos[2]+.08
  for i in range(hold_frame+1,T):
   lift=smooth((i-lift_start)/(lift_end-lift_start));out=smooth((i-lift_end)/(T-1-lift_end));w[i]=(1-out)*(held_pos+[0,0,.08*lift])+out*free;q[i]=held_q;weight=smooth((i-open_start)/(open_end-open_start));f[i]=(1-weight)*held_f+weight*opened
  row[f'robot_{side}_wrist_position']=w.tolist();row[f'robot_{side}_wrist_wxyz']=q.tolist();row[f'robot_{side}_finger_joints']=f.tolist()
 pq.write_table(pa.Table.from_pylist([row]),path,compression='zstd')
 from batch_native import _docker
 image=_docker(['inspect','v2d-batch-episode32-retarget','--format','{{.Image}}']).stdout.strip();_docker(['run','--rm','--name','v2d-brush-safe-release-fk-cpu','--ipc','host','-e','CUDA_VISIBLE_DEVICES=','-e','V2D_TRANSITION_REFERENCE='+REF,'-e','V2D_REBUILD_ONLY=1','-e','V2D_CONTACT_APPLY_OUTPUT=research/brush-safe-release-fk.json','-v',f'{R}:/episode','-v',f'{BASE}/video_to_data/robotic_grounding/source/robotic_grounding/robotic_grounding/assets:/v2d/assets:ro','--entrypoint','/isaac-sim/python.sh',image,'-u','/episode/scripts/apply_collision_transitions.py'],R/'logs/brush-safe-release-fk.log')
 save(R/'research/brush-safe-release-reference.json',{'updated_utc':utc(),'reference':REF,'finger_closure_source_interval':[34,38],'placement_source_interval':[235,245],'placement_uses_tested_initial_brush_in_rack_pose':True,'alternate_symmetric_yaw_branch_chosen':bool(alternate),'right_finger_open_interval':[246,249],'left_finger_open_interval':[247,250],'withdrawal_is_upward_before_outward':True,'rigid_original_meshes_and_materials_preserved':True,'object_assistance':0.})
 release=[{'side':'right','after_source_frame':246,'ramp_source_frames':3,'open_posture_source_frame':262,'zero_wrist_residual':True,'wrist_after_source_frame':245},{'side':'left','after_source_frame':247,'ramp_source_frames':3,'open_posture_source_frame':262,'zero_wrist_residual':True,'wrist_after_source_frame':244}]
 results=[]
 for xy in [25.,100.]:
  status('running',f'Whole-source32-world pickup/release test; left horizontal stiffness{xy:g}N/m')
  left={'kp':25.,'kd':6.,'angular_kp':25.};feedback={'left':2.,'right':.5,'maximum_correction_m':.03,'right_wrist_residual_gain':0.,'right_finger_residual_gain':0.}
  if xy>25:left['axes']={axis:{'kp':xy,'kd':12.} for axis in 'xy'};feedback['left_wrist_position_residual_gains']=[25/xy,25/xy,1.]
  candidate=review(30,45001728,reference_override=REF,root_override=R,container_override='v2d-consistent-episode30-newton',substeps_override=20,recording_world_count=32,tracking_feedback=feedback,wrist_calibration={'left':left,'right':{'kp':500.,'kd':30.,'angular_kp':100.}},release_spec=release)
  accepted=catalog(30,BASE/'runs/episode_000030_newton_batch_20261015',True)['best'];previous=attach(accepted['run'],accepted['selected']);eligible=better(candidate,previous);candidate['passes_current_accepted_policy_gate']=eligible;file=R/f'research/safe-release-brush-left{int(xy)}-review.json';save(file,candidate);results.append({'review':str(file),'eligible':eligible,'mean_iou':candidate['source_mean_object_iou'],'completion_rate':candidate['completion_rate_32_worlds']});print('Safe pickup/release',xy,candidate['source_mean_object_iou'],candidate['completion_rate_32_worlds'],'better',eligible,flush=True)
 save(R/'research/safe-release-brush-completion.json',{'updated_utc':utc(),'results':results});status('completed','Pickup and release ablations checked',eligible_for_independent_publication=any(r['eligible'] for r in results),results=results)
if __name__=='__main__':
 try:main()
 except Exception:status('failed','Pickup/release evaluation failed; previous video retained',error=traceback.format_exc());raise
