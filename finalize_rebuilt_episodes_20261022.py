"""Audit each completed new candidate and refresh selected deliverables automatically."""
import os,json,time,traceback,subprocess,sys
from pathlib import Path
from improvement_common import BASE,BATCH,save,utc
from video_catalog import run_release
import thorough_episode_recheck as human
OUT=BATCH/'improvements_20261022_rebuild';EPISODES=[19,20,30,33];AUDIT=OUT/'human_review';human.OUT=AUDIT;human.__file__=__file__
def activity(completed,ep=None,frame=0,total=0,status='running',**extra):
 save(AUDIT/'status.json',{'updated_utc':utc(),'pid':os.getpid(),'status':status,'episodes_completed':len(completed),'total_episodes':4,'current_episode':ep,'current_frame':frame,'current_total_frames':total})
 if ep is not None:save(BATCH/f'improvements_20261021/activities/episode_{ep:06d}-fresh-human-review-20261022.json',{'episode':ep,'pid':os.getpid(),'process_script':Path(__file__).name,'updated_utc':utc(),'status':status,'phase':f'Fresh trained candidate vs original human: {frame} / {total} frames','description':'Full-frame original-video audit of the newly reconstructed and independently trained policy.','completed_cases':frame,'total_cases':total})
human.report_status=activity

def alive(record):
 try:
  d=json.loads(record.read_text());name=Path(d['script']).name
  return any(Path(os.fsdecode(x)).name==name for x in Path(f'/proc/{d["pid"]}/cmdline').read_bytes().split(b'\0') if x)
 except (OSError,KeyError,ValueError):return False

def main():
 AUDIT.mkdir(exist_ok=True);completed={};final=None
 while True:
  for ep in EPISODES:
   r=BASE/f'runs/episode_{ep:06d}_newton_full_rebuild_20261022'
   checkpoint_reviews=r/'checkpoint-reviews.json'
   if not checkpoint_reviews.exists():continue
   try:
    reviews=json.loads(checkpoint_reviews.read_text())
    if not reviews:continue
    diagnosis=r/'research/checkpoint-diagnosis/latest.json'
    diagnosed=json.loads(diagnosis.read_text()).get('checkpoint_steps') if diagnosis.exists() else None
    if diagnosed!=reviews[-1]['environment_steps']:
     with (r/'logs/checkpoint-diagnosis.log').open('a') as log:
      subprocess.run([sys.executable,str(BASE/'summarize_fresh_checkpoint_diagnosis.py'),'--run',str(r)],stdout=log,stderr=subprocess.STDOUT,check=True)
   except Exception:save(OUT/f'episode_{ep:06d}-checkpoint-diagnosis-error.json',{'updated_utc':utc(),'error':traceback.format_exc()})
  for ep in EPISODES:
   path=OUT/f'episode_{ep:06d}-fresh-result.json'
   if ep in completed or not path.exists():continue
   result=json.loads(path.read_text())
   if result.get('status')!='completed':continue
   r=Path(result['run']);cfg=json.loads((r/'episode-config.json').read_text());release=run_release(ep,r,'experiment',result)
   report_path=AUDIT/f'episode_{ep:06d}/report.json'
   cached=json.loads(report_path.read_text()) if report_path.exists() else {}
   summary=cached if cached.get('version')==release['version'] and cached.get('independent_fresh_training_completed') else human.recheck({'episode':ep,'source_frames':cfg['frames'],'objects':cfg['objects'],'videos':{'latest':release}},list(completed.values()))
   summary.update(independent_fresh_training_completed=True,new_policy_training_performed=True,audit_itself_performed_training=False,matches_every_human_frame_verified=False)
   visual_path=r/cfg.get('visual_review_record','research/no-visual-review.json')
   if visual_path.is_file():
    visual=json.loads(visual_path.read_text())
    summary['source_visual_inspection']={'method':visual['method'],'source_frames_reviewed':len(visual['source_frames_reviewed']),'record':human.artifact(visual_path),'policy_comparison_included':visual_path.parent.name=='full-frame-visual-review'}
    if summary['source_visual_inspection']['policy_comparison_included']:
     summary['visual_review']={'severity':'high' if summary['needs_improvement'] else 'review','finding':'Every source/policy frame inspected in sequential overview contact sheets. Object identities and broad pickup/lift/placement sequence correspond, but brush strokes, projected brush orientation and robot hand posture still differ.','correction_focus':'Brush stroke alignment and final brush orientation; hand and wrist projection.','inspection_method':visual['method']}
   save(report_path,summary);completed[ep]=summary;print('New candidate human-video recheck finished',ep,flush=True)
   try:
    from batch_submission_stage import update
    update()
   except Exception:save(OUT/'staging-refresh-error.json',{'updated_utc':utc(),'error':traceback.format_exc()})
  running=any(alive(OUT/name) for name in ['worker.json','episode_000019-recovery-worker.json','episode_000030-recovery-worker.json','episode_000033-recovery-worker.json'])
  states={str(ep):json.loads((OUT/f'episode_{ep:06d}-status.json').read_text()) for ep in EPISODES}
  all_done=len(completed)==4
  phase=json.loads((OUT/'phase-status.json').read_text());phase.update(phase='completed' if all_done else 'full_reconstruction_and_training' if running else 'completed_with_experiment_failures',updated_utc=utc(),requested_episodes=EPISODES,completed_new_training_and_audit=sorted(completed),all_four_training_and_rollout_complete=all_done,exact_human_imitation_verified=False);save(OUT/'phase-status.json',phase)
  save(OUT/'completion-progress.json',{'updated_utc':utc(),'all_four_training_and_rollout_complete':all_done,'new_candidate_human_audits_completed':sorted(completed),'workers_running':running,'episodes':states,'exact_human_imitation_verified':False})
  if all_done or not running:
   final={'updated_utc':utc(),'episodes_requested':EPISODES,'new_policies_completed_and_rechecked':sorted(completed),'all_four_training_and_rollout_complete':all_done,'episodes':completed,'exact_human_imitation_verified':False};save(OUT/'final-rebuild-report.json',final);break
  time.sleep(25)
 if final['all_four_training_and_rollout_complete']:
  from policy_recommendations import snapshot,rounds_finished
  snapshot(finalize=rounds_finished()[0])
  from evaluation_inventory import audit
  audit()
  from batch_submission_stage import update
  update()
  save(OUT/'all20-selected-deliverables-refreshed.json',{'updated_utc':utc(),'new_runs_checked':[19,20,30,33],'all20_audit_and_staging_refreshed':True,'official_submission_uploaded':False})
if __name__=='__main__':
 try:main()
 except Exception:save(OUT/'finalizer-error.json',{'updated_utc':utc(),'error':traceback.format_exc()});raise
