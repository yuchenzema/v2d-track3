"""Run the isolated episode 20 grasp reconstruction and backward pickup experiment."""
import concurrent.futures
import fcntl
import json
import time
import traceback
from improvement_common import BATCH,save,utc
from fresh_improvement_run import run

OUT=BATCH/'improvements_20261020_backward'

if __name__=='__main__':
    OUT.mkdir(exist_ok=True)
    with (OUT/'queue.lock').open('a') as lock:
        fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
        results={};pending={};launched=set()
        try:
            with concurrent.futures.ThreadPoolExecutor(max_workers=3) as pool:
                while True:
                    registry=OUT/'episodes.json'
                    episodes=json.loads(registry.read_text())['episodes'] if registry.exists() else [20]
                    for ep in episodes:
                        assert ep in [19,20,30,32,33]
                        if ep in launched:continue
                        existing=OUT/f'episode_{ep:06d}-fresh-result.json'
                        result=json.loads(existing.read_text()) if existing.exists() else {}
                        launched.add(ep)
                        if result.get('status')=='completed':results[ep]=result
                        else:pending[pool.submit(run,ep,'backward_grasp_20261020')]=ep
                    for future in [f for f in pending if f.done()]:
                        ep=pending.pop(future)
                        try:results[ep]=future.result()
                        except Exception as error:results[ep]={'episode':ep,'status':'failed','error':str(error)}
                    save(OUT/'phase-status.json',{'phase':'backward_grasp_training' if pending else
                        'completed_with_experiment_failures' if any(r['status']=='failed' for r in results.values()) else 'completed',
                        'updated_utc':utc(),'episodes':episodes,
                        'training_method':'Holding, progressively earlier pickup starts, then full source motion',
                        'evaluation_object_assistance':0.,'full_human_motion_match_verified':False,
                        'completed':[e for e,r in results.items() if r['status']=='completed'],
                        'failed':[e for e,r in results.items() if r['status']=='failed'],
                        'measured_improvements':[e for e,r in results.items() if r.get('measured_improvement')],
                        'unresolved':[e for e,r in results.items() if not r.get('measured_improvement')]})
                    if not pending:break
                    time.sleep(15)
        except BaseException:
            save(OUT/'phase-status.json',{'phase':'failed','updated_utc':utc(),'episodes':[20],
                'error':traceback.format_exc(),'verified_policy_improvement':False})
            raise
