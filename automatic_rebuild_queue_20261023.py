"""Persistent staged queue with bounded retries, honest status and GUI recovery."""
import fcntl
import hashlib
import json
import os
import shlex
from pathlib import Path
import subprocess
import time
import urllib.request
from improvement_common import BASE, BATCH, save, utc

OUT=BATCH/'improvements_20261023_rebuild'
EPISODES=[19,20,30,33]
PY=BASE/'runs/episode_000005_newton_20261015/.venv-video/bin/python'
MAX_FAILURES=3

def read(path,default=None):
    try:return json.loads(path.read_text())
    except (OSError,ValueError):return default

def root(ep):return BASE/f'runs/episode_{ep:06d}_newton_full_rebuild_20261023'

def fingerprint():
    paths=[BASE/'rerun_four_episodes_20261023.py',BASE/'refine_initial_rest_20261022.py',
           BASE/'rebuild_full_physics_preflight_20261022.py',BASE/'rebuild_release_targets_20261022.py',
           *BASE.glob('*_20261023.py')]
    return hashlib.sha256(b''.join(p.read_bytes() for p in sorted(set(paths)))).hexdigest()

def completed(ep):
    result=read(OUT/f'episode_{ep:06d}-fresh-result.json',{})
    report=read(OUT/f'human_review/episode_{ep:06d}/report.json',{})
    status=read(root(ep)/'research/rebuild-status.json',{})
    return (result.get('status')=='completed' and result.get('artifact_integrity',{}).get('integrity_passed')
            and report.get('independent_fresh_training_completed')
            and status.get('status')=='completed')

def next_stage(ep):
    if completed(ep):return None
    return 'train_rollout' if (root(ep)/'research/full-reconstruction-completed.json').exists() else 'reconstruct'

def ensure_gui():
    unit=Path.home()/'.config/systemd/user/v2d-track3-gui.service'
    if unit.exists():
        subprocess.run(['systemctl','--user','start','v2d-track3-gui.service'],check=True)
        return
    try:
        with urllib.request.urlopen('http://127.0.0.1:8768/api/status',timeout=10) as response:
            if response.status==200:return
    except OSError:pass
    meta=read(BATCH/'progress-dashboard-worker.json',{})
    try:
        argv=Path(f'/proc/{meta["pid"]}/cmdline').read_bytes().split(b'\0')
        if str(BASE/'progress_dashboard.py').encode() in argv:return
    except (OSError,KeyError):pass
    process=subprocess.Popen([str(BASE/'.venv/bin/python'),'-u',str(BASE/'progress_dashboard.py')],cwd=BASE,
                             stdout=(BATCH/'progress-dashboard.log').open('a'),stderr=subprocess.STDOUT,start_new_session=True)
    save(BATCH/'progress-dashboard-worker.json',{'pid':process.pid,'script':'progress_dashboard.py','updated_utc':utc(),'url':'http://127.0.0.1:8768/'})

def publish(state):
    state.update(updated_utc=utc(),pid=os.getpid(),process_script=Path(__file__).name)
    state['episodes']={str(ep):{'next_stage':next_stage(ep),'completed':bool(completed(ep)),
                                'state':read(root(ep)/'research/rebuild-status.json',{})} for ep in EPISODES}
    save(OUT/'automatic-queue.json',state)
    phase=read(OUT/'phase-status.json',{})
    phase.update(phase='completed' if state['status']=='completed' else 'full_reconstruction_and_training',
                 updated_utc=utc(),requested_episodes=EPISODES,automatic_queue=state,
                 all_four_training_and_rollout_complete=state['status']=='completed',exact_human_imitation_verified=False)
    save(OUT/'phase-status.json',phase)

def recover_containers():
    """Stop orphaned diagnostics; preserve and adopt any existing learner."""
    def docker(args):
        return subprocess.check_output(['sg','docker','-c',shlex.join(['docker',*args])],text=True)
    ids=docker(['ps','-q']).split();actions=[]
    if not ids:return
    for info in json.loads(docker(['inspect',*ids])):
        if not any(m.get('Source') in {str(root(ep)) for ep in EPISODES} for m in info.get('Mounts',[])):continue
        top=docker(['top',info['Id'],'-eo','pid,args'])
        learner=any('/episode/scripts/train_coherent_actor_transfer.py' in line for line in top.splitlines())
        if not learner:docker(['stop','-t','10',info['Id']])
        actions.append({'container':info['Name'],'action':'adopt existing learner' if learner else 'stop orphan diagnostics; restart from durable stage'})
    save(OUT/'service-container-recovery.json',{'updated_utc':utc(),'actions':actions})

def main():
    OUT.mkdir(exist_ok=True)
    recover_containers()
    state=read(OUT/'automatic-queue.json',{})
    state.update(status='running',active=None,retry_policy={'maximum_consecutive_failures_per_stage':MAX_FAILURES,
                 'backoff_seconds':[30,120,600],'repaired_code_requeues_failed_stages':True})
    state.setdefault('attempts',{});state.setdefault('history',[])
    previous_fingerprint=state.get('code_fingerprint')
    while True:
        ensure_gui()
        current_fingerprint=fingerprint()
        if previous_fingerprint and previous_fingerprint!=current_fingerprint:
            state['history'].append({'utc':utc(),'event':'Code repair detected; failed stages requeued'})
            state['attempts']={}
        state['code_fingerprint']=previous_fingerprint=current_fingerprint
        pending=[(ep,next_stage(ep)) for ep in EPISODES if next_stage(ep)]
        if not pending:
            state.update(status='completed',active=None);publish(state)
            from policy_recommendations import snapshot,rounds_finished
            snapshot(finalize=rounds_finished()[0])
            from batch_submission_stage import update
            update()
            return
        ready=[]
        for ep,stage in pending:
            retry=state['attempts'].get(f'{ep}:{stage}',{})
            if retry.get('failures',0)<MAX_FAILURES and retry.get('retry_at',0)<=time.time():ready.append((ep,stage))
        # Reconstruct all scenes first, reserving GPU memory for vision and
        # reference validation. Then train and export one episode at a time.
        ready.sort(key=lambda item:(item[1]!='reconstruct',EPISODES.index(item[0])))
        if not ready:
            state.update(status='needs_repair' if all(state['attempts'].get(f'{ep}:{stage}',{}).get('failures',0)>=MAX_FAILURES for ep,stage in pending) else 'retry_wait',active=None)
            publish(state);time.sleep(30);continue
        ep,stage=ready[0];key=f'{ep}:{stage}'
        state.update(status='running',active={'episode':ep,'stage':stage,'started_utc':utc()})
        log=OUT/f'episode_{ep:06d}-{stage}-queue.log'
        with log.open('a') as output:
            process=subprocess.Popen([str(PY),'-u',str(BASE/'run_queued_episode_20261023.py'),str(ep),stage],cwd=BASE,stdout=output,stderr=subprocess.STDOUT)
            state['active']['pid']=process.pid
            save(OUT/'worker.json',{'pid':process.pid,'script':'run_queued_episode_20261023.py','episode':ep,'stage':stage,'started_utc':utc()})
            while process.poll() is None:
                publish(state);time.sleep(10)
            code=process.returncode
        retry=state['attempts'].setdefault(key,{'failures':0})
        if code:
            retry['failures']+=1;retry['retry_at']=time.time()+[30,120,600][min(retry['failures']-1,2)]
            retry['last_error']=read(root(ep)/'research/queue-stage-error.json',{})
            state['history'].append({'utc':utc(),'episode':ep,'stage':stage,'exit_code':code,'consecutive_failures':retry['failures']})
        else:
            retry.update(failures=0,retry_at=0)
            state['history'].append({'utc':utc(),'episode':ep,'stage':stage,'exit_code':0})
        state['active']=None;publish(state)

if __name__=='__main__':
    OUT.mkdir(exist_ok=True)
    with (OUT/'automatic-queue.lock').open('a') as lock:
        fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB);main()
