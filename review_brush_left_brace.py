"""Isolate left wrist stiffness and learned position residuals on the same physical pickup."""
import os,traceback
from improvement_common import BASE,BATCH,save,utc,better
from batch_review import review
from video_catalog import catalog
from object_motion_quality import attach
R=BASE/'runs/episode_000030_newton_consistent_brush_20261021'
A=BATCH/'improvements_20261021/activities/episode_000030-left-brace.json'
def status(state,phase,**extra):
 save(A,{'episode':30,'pid':os.getpid(),'process_script':os.path.basename(__file__),'updated_utc':utc(),'status':state,'phase':phase,'description':'Isolating rack drift: compare left wrist stiffness and learned horizontal residuals, retaining the same brush pickup, object geometry, source timing and force limits.',**extra})
def main():
 release=[{'side':'right','after_source_frame':246,'ramp_source_frames':3,'open_posture_source_frame':262,'zero_wrist_residual':True,'wrist_after_source_frame':245},{'side':'left','after_source_frame':247,'ramp_source_frames':3,'open_posture_source_frame':262,'zero_wrist_residual':True,'wrist_after_source_frame':244}]
 accepted=catalog(30,BASE/'runs/episode_000030_newton_batch_20261015',True)['best'];previous=attach(accepted['run'],accepted['selected']);results=[]
 for xy,gain in [(25.,0.),(100.,0.),(100.,1.)]:
  status('running',f'Checking rack brace: horizontal stiffness {xy:g} N/m; actor position gain {gain:g}',completed_cases=len(results),total_cases=3)
  left={'kp':25.,'kd':6.,'angular_kp':25.}
  if xy>25:left['axes']={axis:{'kp':xy,'kd':12.} for axis in 'xy'}
  feedback={'left':2.,'right':.5,'maximum_correction_m':.03,'right_wrist_residual_gain':0.,'right_finger_residual_gain':0.,'left_wrist_position_residual_gains':[gain,gain,1.]}
  c=review(30,45001728,reference_override='track3_brush_safe_release',root_override=R,container_override='v2d-consistent-episode30-newton',substeps_override=20,recording_world_count=32,tracking_feedback=feedback,wrist_calibration={'left':left,'right':{'kp':500.,'kd':30.,'angular_kp':100.}},release_spec=release)
  eligible=better(c,previous);c['passes_current_accepted_policy_gate']=eligible;p=R/f'research/left-brace-{int(xy)}-gain{gain:g}-review.json';save(p,c)
  result={'review':str(p),'eligible':eligible,'mean_iou':c['source_mean_object_iou'],'completion_rate':c['completion_rate_32_worlds']};results.append(result);print(result,flush=True)
 status('completed','Rack-brace ablations checked',eligible_for_independent_publication=any(r['eligible'] for r in results),results=results)
if __name__=='__main__':
 try:main()
 except Exception:status('failed','Rack-brace diagnostic failed',error=traceback.format_exc());raise
