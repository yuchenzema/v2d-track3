"""Retain improved held motion while handing placement back to the learned actor."""
import os,json,traceback
from improvement_common import BASE,BATCH,save,utc,better
from batch_review import review
from video_catalog import catalog
R=BASE/'runs/episode_000032_newton_backward_grasp_20261020'
activity=BATCH/'improvements_20261021/activities/episode_000032-feedback-release.json'
def status(state,phase,**extra):
 save(activity,{'episode':32,'pid':os.getpid(),'process_script':__file__.split('/')[-1],'updated_utc':utc(),'status':state,'phase':phase,'description':'Checking held-object feedback with an earlier handoff to the learned release motion; no object assistance.',**extra})
def main():
 results=[]
 for lead in [8,16]:
  status('running',f'Uniform32-world controller test; feedback fades{lead} source frames before release')
  profile={'left':2.,'right':2.,'maximum_correction_m':.015,'release_lead_source_frames':lead}
  row=review(32,60000256,root_override=R,container_override='v2d-backward-episode32-newton',substeps_override=20,recording_world_count=32,tracking_feedback=profile)
  accepted=catalog(32,BASE/'runs/episode_000032_newton_batch_20261015',True)['best']['selected'];eligible=better(row,accepted);row['passes_current_accepted_policy_gate']=eligible
  path=R/f'research/robot-tracking-grid-20261021/release-lead{lead}-review.json';save(path,row);results.append({'review':str(path),'eligible':eligible})
  print('Release lead',lead,'meanIoU',row['source_mean_object_iou'],'nativeSR',row['completion_rate_32_worlds'],'better',eligible,flush=True)
 save(R/'research/robot-tracking-grid-20261021/release-completion.json',{'updated_utc':utc(),'results':results});status('completed','Controller release checks finished; publication requires independent128-world validation',results=results)
if __name__=='__main__':
 try:main()
 except Exception:status('failed','Controller checks failed; previous video retained',error=traceback.format_exc());raise
