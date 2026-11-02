"""Run reviewed episode10/16 contact finetuning as GPU capacity becomes available."""
import concurrent.futures,fcntl,json,time,traceback
from improvement_common import BASE,BATCH,save,utc
from fresh_improvement_run import run
OUT=BATCH/'improvements_20261020_contacts';EPISODES=[16,10]
def reserve_next_available_slot():
 """Give reviewed contact jobs the next slot without interrupting learners."""
 path=BATCH/'training-priority.json'
 if not path.exists():return
 priority=json.loads(path.read_text());waiting=[]
 for ep in EPISODES:
  root=BASE/'runs'/f'episode_{ep:06d}_newton_grasp_curriculum_20261020'
  decision=root/'research/physical-grasp-reference-selection.json'
  selection=json.loads(decision.read_text()) if decision.exists() else {}
  result=OUT/f'episode_{ep:06d}-fresh-result.json';finished=json.loads(result.read_text()).get('status') in ['completed','failed'] if result.exists() else False
  if selection.get('approved_for_independent_training') and not (root/'coherent-input-checksums.json').exists() and not finished:waiting.append(ep)
 deferred=priority.get('deferred_for_contact_admission_run_names',[])
 names=list(dict.fromkeys(priority.get('allowed_run_names',[])+deferred))
 deferred=[n for n in names if any(n.startswith(f'episode_{ep:06d}_') for ep in [19,33])] if waiting else []
 priority.update(allowed_run_names=[n for n in names if n not in deferred],
  deferred_for_contact_admission_run_names=deferred,contact_episodes_waiting_for_next_slot=waiting,updated_utc=utc())
 save(path,priority)
def experiment(ep):
 for attempt in range(2):
  try:
   result=run(ep,'grasp_curriculum_20261020');save(OUT/f'episode_{ep:06d}-fresh-result.json',result);return result
  except Exception:
   print('Contact finetuning attempt failed',ep,attempt+1,traceback.format_exc(),flush=True)
 result={'episode':ep,'status':'failed','updated_utc':utc(),'error':'Both bounded training/export attempts failed; see worker log'}
 save(OUT/f'episode_{ep:06d}-fresh-result.json',result);return result
if __name__=='__main__':
 OUT.mkdir(exist_ok=True)
 with (OUT/'queue.lock').open('a') as lock:
  fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB);launched=set();results={};pending={}
  with concurrent.futures.ThreadPoolExecutor(max_workers=2) as pool:
   while True:
    review=[]
    for ep in EPISODES:
     if ep in launched:continue
     root=BASE/'runs'/f'episode_{ep:06d}_newton_grasp_curriculum_20261020';path=root/'research/physical-grasp-reference-selection.json'
     selection=json.loads(path.read_text()) if path.exists() else {}
     if not selection.get('approved_for_independent_training') or selection.get('training_curriculum')!='contact_finetune':review.append(ep);continue
     launched.add(ep);existing=OUT/f'episode_{ep:06d}-fresh-result.json'
     old=json.loads(existing.read_text()) if existing.exists() else {}
     if old.get('status')=='completed':results[ep]=old
     else:pending[pool.submit(experiment,ep)]=ep
    for future in [f for f in pending if f.done()]:
     ep=pending.pop(future)
     try:results[ep]=future.result()
     except Exception as error:results[ep]={'episode':ep,'status':'failed','error':str(error),'updated_utc':utc()};save(OUT/f'episode_{ep:06d}-fresh-result.json',results[ep])
    reserve_next_available_slot()
    finished=len(results)==len(EPISODES)
    save(OUT/'phase-status.json',{'phase':('completed_with_experiment_failures' if any(r['status']=='failed' for r in results.values()) else 'completed') if finished else
     'contact_finetuning' if pending else 'contact_model_review','updated_utc':utc(),'episodes':EPISODES,
     'held_for_reconstruction_review':review,'running_or_queued':list(pending.values()),
     'completed':[e for e,r in results.items() if r['status']=='completed'],'failed':[e for e,r in results.items() if r['status']=='failed'],
     'measured_improvements':[e for e,r in results.items() if r.get('measured_improvement')],
     'evaluation_object_assistance':0.,'full_human_motion_match_verified':False,
     'method':'Reviewed cup nesting and spoon insertion/release; existing episode actor finetuned with zero assistance and no reset settling.'})
    if finished:break
    time.sleep(15)
