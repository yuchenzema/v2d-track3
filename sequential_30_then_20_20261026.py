"""Durable ordered queue: finish episode30 export/audit, then rebuild episode20."""
import fcntl
import json
import os
from pathlib import Path
import subprocess
import time
import traceback
from improvement_common import BASE,BATCH,save,utc

OUT=BATCH/'requested_episodes_20_30'
REQUEST=OUT/'episode20-after30-request.json'
PY=BASE/'runs/episode_000005_newton_20261015/.venv-video/bin/python'
PREDECESSOR=BATCH/'episode30_pickup_repair_20261026'

def read(path,default=None):
    try:return json.loads(path.read_text())
    except (OSError,ValueError):return {} if default is None else default

def predecessor_finished():
    status=read(PREDECESSOR/'status.json')
    result=read(PREDECESSOR/'episode_000030-fresh-result.json')
    review=read(PREDECESSOR/'human_review/episode_000030/report.json')
    return bool(status.get('status')=='completed' and result.get('status')=='completed'
                and result.get('artifact_integrity',{}).get('integrity_passed')
                and review.get('independent_fresh_training_completed'))

def publish(state,reason,**extra):
    row=read(OUT/'status.json')
    row.update(status=state,reason=reason,active=state in ['waiting_episode30','running_episode20'],
               updated_utc=utc(),pid=os.getpid(),process_script=Path(__file__).name,
               execution_order=[30,20],blockers=[],exact_human_match_verified=False,**extra)
    row['new_reconstruction_started']=True
    row['new_training_started']=True
    row['new_rollout_started']=predecessor_finished()
    row.pop('last_execution_check',None)
    episodes=row.setdefault('episodes',{})
    if '20' in episodes:
        stages=episodes['20'].setdefault('stages',{})
        stages.update({name:('queued_after_episode30' if state=='waiting_episode30' else
                            'completed' if state in ['completed_local_checks','needs_improvement'] else
                            'running_or_pending' if state=='running_episode20' else state)
                       for name in ['rebuild','training','rollout']})
        if state=='running_episode20':
            request=read(REQUEST)
            run=BASE/f"runs/episode_000020_newton_human_match_20261023_g{request['generation']:03d}"
            rebuilt=(run/'research/full-reconstruction-completed.json').exists()
            trained=(run/'flash_chord_coherent_actor_transfer/TRAINING_COMPLETED').exists()
            stages.update(rebuild='completed' if rebuilt else 'running',
                          training='completed' if trained else 'running' if rebuilt else 'queued',
                          rollout='running' if trained else 'queued')
    if '30' in episodes:
        episodes['30']['stages']={'rebuild':'completed','training':'completed' if predecessor_finished() else 'running',
                                  'rollout':'completed' if predecessor_finished() else 'pending'}
    save(OUT/'status.json',row)

def main():
    OUT.mkdir(exist_ok=True)
    with (OUT/'sequential-worker.lock').open('a') as lock:
        fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
        while not predecessor_finished():
            if read(BATCH/'user-pause.json').get('status')=='paused_by_user':
                publish('paused_by_user','Queue paused by user; no episode20 launch.');return
            publish('waiting_episode30','Episode30 training, rollout and audit run first. Episode20 reconstruction, training and rollout are queued automatically.')
            time.sleep(15)
        while subprocess.run(['systemctl','--user','is-active','--quiet','v2d-episode30-pickup-repair.service']).returncode==0:
            time.sleep(5)
        if read(BATCH/'user-pause.json').get('status')=='paused_by_user':
            publish('paused_by_user','Queue paused by user; no episode20 launch.');return
        result_path=REQUEST.with_suffix('.result.json')
        for attempt in range(3):
            if result_path.exists():break
            publish('running_episode20','Episode30 pipeline finished. Episode20: reconstructing from original cameras, training from frame0, then full-source rollout and strict correspondence audit.',attempt=attempt+1)
            with (OUT/'episode20-worker.log').open('a') as log:
                code=subprocess.run([str(PY),'-u',str(BASE/'execute_human_match_job_20261023.py'),'--request',str(REQUEST)],
                                    cwd=BASE,stdout=log,stderr=subprocess.STDOUT).returncode
            if code==0 and result_path.exists():break
            if attempt==2:raise RuntimeError('Episode20 pipeline failed after three attempts; see episode20-worker.log')
            time.sleep(30)
        result=read(result_path);strict=result.get('strict_check',{})
        passed=bool(strict.get('passes_strict_local_checks'))
        publish('completed_local_checks' if passed else 'needs_improvement',
                'Episode20 pipeline completed. Strict local checks passed; exact human matching remains unverified.' if passed else
                'Episode20 pipeline completed, but source motion checks still fail. Candidate retained for inspection; current accepted video preserved.',
                episode20_result=result,all_pipeline_stages_executed=result.get('all_pipeline_stages_executed',False))

if __name__=='__main__':
    try:main()
    except Exception:
        publish('failed','Ordered queue needs repair; saved checkpoints preserved.',error=traceback.format_exc())
        raise
