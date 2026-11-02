"""Validate the prepared contact-calibrated reference with verified motor gains."""
import os,json,traceback
from improvement_common import BASE,BATCH,save,utc,better
from batch_review import review
from video_catalog import catalog
R=BASE/'runs/episode_000030_newton_consistent_brush_20261021';A=BATCH/'improvements_20261021/activities/episode_000030-calibrated-review-retry.json'
def status(state,phase,**extra):save(A,{'episode':30,'pid':os.getpid(),'process_script':__file__.split('/')[-1],'updated_utc':utc(),'status':state,'phase':phase,'description':'Whole-source hybrid rollout using the retargeted right-hand trajectory, trained left-hand actor, compliant bristle tips and verified actual wrist gains. No object guidance or resets.',**extra})
def main():
 status('running','Native32-world continuous calibrated rollout; motor gains asserted at runtime')
 row=review(30,45001728,reference_override='track3_brush_grip_calibrated',root_override=R,container_override='v2d-consistent-episode30-newton',substeps_override=20,recording_world_count=32,tracking_feedback={'left':2.,'right':.5,'maximum_correction_m':.015,'right_wrist_residual_gain':0.,'right_finger_residual_gain':0.},bristle_compliance={'stiffness':20.,'damping':1.,'tip_max_z':-.015},wrist_calibration={'left':{'kp':25.,'kd':6.,'angular_kp':25.},'right':{'kp':500.,'kd':30.,'angular_kp':100.}})
 accepted=catalog(30,BASE/'runs/episode_000030_newton_batch_20261015',True)['best']['selected'];eligible=better(row,accepted);row['passes_current_accepted_policy_gate']=eligible;save(R/'research/calibrated-brush-policy-review.json',row);status('completed','Calibrated continuous rollout checked; previous video retained unless all promotion gates pass',eligible_for_independent_publication=eligible,source_mean_object_iou=row['source_mean_object_iou'],completion_rate=row['completion_rate_32_worlds']);print('Calibrated rollout meanIoU',row['source_mean_object_iou'],'nativeSR',row['completion_rate_32_worlds'],'better',eligible,flush=True)
if __name__=='__main__':
 try:main()
 except Exception:status('failed','Calibrated rollout failed; previous video retained',error=traceback.format_exc());raise
