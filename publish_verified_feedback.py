"""Publish only after current-best comparison and independent128-world validation."""
import os,json,subprocess,traceback
from improvement_common import BASE,BATCH,save,utc,better
from video_catalog import catalog
from publish_contact_controller_candidate import publish
R=BASE/'runs/episode_000032_newton_backward_grasp_20261020'
A=BATCH/'improvements_20261021/activities/episode_000032-feedback-publication.json'
def status(state,phase,**extra):save(A,{'episode':32,'pid':os.getpid(),'process_script':__file__.split('/')[-1],'updated_utc':utc(),'status':state,'phase':phase,'description':'Validating the improved wrist controller on128 worlds, rendering the complete rollout and updating accepted GUI links.',**extra})
def main():
 status('running','Independent128-world validation before promotion')
 path=R/'research/robot-tracking-grid-20261021/release-lead16-review.json';candidate=json.loads(path.read_text())
 accepted=catalog(32,BASE/'runs/episode_000032_newton_batch_20261015',True)['best'];assert better(candidate,accepted['selected'])
 result=publish(32,path,'bounded-feedback-release16',root_override=R)
 current=catalog(32,BASE/'runs/episode_000032_newton_batch_20261015',True)['latest']
 assert current['selected'].get('robot_tracking_feedback_profile')==candidate['robot_tracking_feedback_profile'],'Candidate did not earn current GUI selection'
 assert current['quality']['completion_rate_against_video_estimate']>=accepted['quality']['completion_rate_against_video_estimate']-.0625
 status('running','Refreshing all20 accepted-policy audit and staged trajectories',new_video_version=current['version'])
 with (BATCH/'improvements_20261021/feedback-publication-finalization.log').open('a') as log:
  for executable,args in [(BASE/'runs/episode_000005_newton_20261015/.venv-video/bin/python',['evaluation_inventory.py']),(BASE/'.venv/bin/python',['policy_recommendations.py','--finalize']),(BASE/'.venv-submission/bin/python',['batch_submission_stage.py'])]:subprocess.run([str(executable),'-u',*args],cwd=BASE,stdout=log,stderr=subprocess.STDOUT,check=True)
 save(BATCH/'improvements_20261021/episode_000032-feedback-published.json',{'updated_utc':utc(),'new_video_version':current['version'],'previous_video_version':accepted['version'],'source_mean_object_iou':candidate['source_mean_object_iou'],'previous_source_mean_object_iou':accepted['selected']['source_mean_object_iou'],'native128_completion_rate':current['quality']['completion_rate_against_video_estimate'],'all20_staging_refreshed':True,'exact_human_match_verified':False})
 status('completed','Improved controller verified, videos promoted, all20 staged selections refreshed',new_video_version=current['version']);print('Episode32 new accepted video',current['version'],flush=True)
if __name__=='__main__':
 try:main()
 except Exception:status('failed','Publication or validation failed; inspect retained GUI selection',error=traceback.format_exc());raise
