"""Prioritize grasp-first retraining for the five still-unsolved episodes."""
from pathlib import Path
import concurrent.futures
import fcntl
import json
import time
import traceback
from improvement_common import BASE,BATCH,save,utc
from fresh_improvement_run import run,prepare

OUT=BATCH/'improvements_20261020_grasp'
PRIORITY=[20,32,19,30,33]
VARIANT='grasp_curriculum_20261020'


def phase(name,**kwargs):
    save(OUT/'phase-status.json',{'phase':name,'updated_utc':utc(),'episodes':PRIORITY,
         'maximum_concurrent_priority_learners':3,
         'training_method':'Opening-grasp pretraining, then the full unchanged source motion',
         'prefix_training_budget':15_000_576,'full_motion_training_budget':50_000_896,
         'final_evaluation_uses_full_episode':True,'evaluation_object_assistance':0.,**kwargs})


def experiment(ep):
    p=OUT/f'episode_{ep:06d}-fresh-result.json'
    if p.exists() and json.loads(p.read_text()).get('status')=='completed':return json.loads(p.read_text())
    for attempt in range(2):
        try:return run(ep,VARIANT)
        except Exception:
            print('Grasp curriculum failure',ep,'attempt',attempt+1,traceback.format_exc(),flush=True)
    return json.loads(p.read_text())


def watch():
    for ep in PRIORITY:prepare(ep,VARIANT)
    reservation=BATCH/'training-priority.json'
    held=json.loads(reservation.read_text()).get('held_for_reconstruction_review',[]) if reservation.exists() else []
    additional=json.loads(reservation.read_text()).get('additional_allowed_run_names',[]) if reservation.exists() else []
    priority={'active':True,'updated_utc':utc(),'episodes':PRIORITY,
        'maximum_concurrent_learners':3,'extra_learner_requires_gpu_reserve_mb':3500,
        'extra_learner_requires_available_ram_gib':16,
        'allowed_run_names':[f'episode_{ep:06d}_newton_{VARIANT}' for ep in PRIORITY if ep not in held]+additional,
        'additional_allowed_run_names':additional,
        'held_for_reconstruction_review':held,
        'preserve_running_episodes':[10,16],
        'reason':'The five episodes explicitly reported as unimproved run ahead of non-priority experiments'}
    save(BATCH/'training-priority.json',priority)
    results={}
    # Waiting reconstruction reviews must not block another ready episode.
    # GPU admission still enforces the independent three-learner memory cap.
    with concurrent.futures.ThreadPoolExecutor(max_workers=len(PRIORITY)) as pool:
        pending={pool.submit(experiment,ep):ep for ep in PRIORITY}
        while pending:
            done,_=concurrent.futures.wait(pending,timeout=30,return_when=concurrent.futures.FIRST_COMPLETED)
            for future in done:
                ep=pending.pop(future)
                try:results[ep]=future.result()
                except Exception as error:
                    results[ep]={'episode':ep,'status':'failed','error':str(error),'updated_utc':utc()}
                    save(OUT/f'episode_{ep:06d}-fresh-result.json',results[ep])
            phase('grasp_curriculum_training',completed=[ep for ep,row in results.items() if row['status']=='completed'],
                  failed=[ep for ep,row in results.items() if row['status']=='failed'],
                  measured_improvements=[ep for ep,row in results.items() if row.get('measured_improvement')])
    priority.update(active=False,updated_utc=utc());save(BATCH/'training-priority.json',priority)
    phase('completed',results=results,measured_improvements=[ep for ep,row in results.items() if row.get('measured_improvement')],
          unresolved=[ep for ep,row in results.items() if not row.get('measured_improvement')],
          exact_human_motion_match_verified=False)


if __name__=='__main__':
    OUT.mkdir(exist_ok=True)
    with (OUT/'queue.lock').open('a') as lock:
        fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
        try:watch()
        except Exception as error:
            phase('failed',error=str(error),traceback=traceback.format_exc());raise
