"""Independent, lightweight health checks for the serial challenge queue.

Never launch episode workers or terminate learners. Restart an unavailable
controller only when no episode owner or protected learner is still alive.
"""
from datetime import datetime, timezone
import fcntl
import json
import os
from pathlib import Path
import shutil
import signal
import re
import subprocess
import time

from improvement_common import BASE, BATCH, save, utc
from episode_execution_guard import owner_alive, available_ram_bytes, LOCK
from episode_completion_gate import completion

OUT = Path(os.environ.get('V2D_QUEUE_DIR',BATCH / 'human_match_loop_20261023'))
EXPECTED = {1, 3, 5, 6, 8, 10, 14, 15, 16, 19, 20, 25, 28, 30, 32, 33, 35, 36, 37, 38}
if os.environ.get('V2D_FOCUS_EPISODES'):EXPECTED={int(value) for value in os.environ['V2D_FOCUS_EPISODES'].split(',')}
CONTROLLER = os.environ.get('V2D_QUEUE_SERVICE','v2d-track3-human-match.service')
GUI = 'v2d-track3-gui.service'


def read(path, default=None):
    try:
        return json.loads(path.read_text())
    except (OSError, ValueError):
        return default


def age(value, now):
    try:
        return max(0, now - datetime.fromisoformat(value.replace('Z', '+00:00')).timestamp())
    except (TypeError, ValueError, AttributeError):
        return None


def latest_metric(path):
    try:
        with path.open('rb') as stream:
            stream.seek(max(0, path.stat().st_size - 65536))
            lines = stream.read().decode(errors='replace').splitlines()
        for line in reversed(lines):
            try:
                value = json.loads(line)
                if 'environment_steps' in value:
                    return value
            except ValueError:
                continue
    except OSError:
        pass
    return {}


def native_learners():
    result = []
    for path in Path('/proc').glob('[0-9]*/cmdline'):
        try:
            args = path.read_bytes().split(b'\0')
            if Path(args[0].decode()).name.startswith('python') and any(
                arg.endswith(b'/train_coherent_actor_transfer.py') for arg in args
            ):
                result.append(int(path.parent.name))
        except (OSError, UnicodeError, ValueError):
            continue
    return sorted(result)


def recovery_decision(service, paused, done, owned_process_alive):
    return (service in {'inactive', 'failed'} and not paused and not done
            and not owned_process_alive)


def source_alignment_idle_problem(state,owned,paused):
    if not paused and not owned and state.get('status')=='source_alignment_repairs_required':
        return 'No episode worker is active: existing source-alignment repairs are exhausted; a verified algorithm/source repair is required.'
    return None


def public_validation_health(validation,owned,paused=False):
    """Distinguish a live service from a public pipeline that cannot advance."""
    if paused:return 'paused_by_user',None
    status=validation.get('status')
    if not owned and status=='failed':
        episode=validation.get('active_episode')
        detail=(validation.get('error') or validation.get('phase') or 'Unknown stage failure').strip().splitlines()[-1]
        return 'public_validation_failed',f'Public episode {episode} cannot advance: {detail}'
    if not owned and status=='public_reference_validation_quality_insufficient':
        return 'public_reference_quality_blocked','Public reconstruction references remain below admission criteria; evaluation training is blocked.'
    if not owned and status=='awaiting_safe_controller_reload':
        return 'public_controller_reload_pending',None
    return 'public_pipeline_validation',None


def strict_progress(state, expected):
    checks={ep:completion(state.get('episodes',{}).get(str(ep),{})) for ep in expected}
    verified={ep for ep,check in checks.items() if check['episode_done']}
    claimed=set(state.get('episodes_done',[]))
    return sorted(verified),sorted(claimed-verified),checks


def service_state(name):
    result = subprocess.run(['systemctl', '--user', 'show', name, '-p', 'ActiveState', '--value'],
                            capture_output=True, text=True, timeout=10)
    return result.stdout.strip() or 'unknown'


def acquire_update_lock(lock,wait_seconds=20.):
    """Wait in the kernel for a short episode boundary, with a bounded deadline."""
    if wait_seconds<=0:raise ValueError('Queue update wait must have a positive deadline')
    def deadline(signum,frame):raise TimeoutError('Episode still owns the queue slot')
    previous_handler=signal.signal(signal.SIGALRM,deadline)
    previous_timer=signal.setitimer(signal.ITIMER_REAL,wait_seconds)
    started=time.monotonic()
    try:
        fcntl.flock(lock,fcntl.LOCK_EX)
        return True
    except TimeoutError:
        return False
    finally:
        signal.setitimer(signal.ITIMER_REAL,0.)
        signal.signal(signal.SIGALRM,previous_handler)
        if previous_timer[0]>0:
            signal.setitimer(signal.ITIMER_REAL,max(.001,previous_timer[0]-(time.monotonic()-started)),previous_timer[1])


def apply_pending_update(paused, done):
    """Reload queue code only between episodes, holding the global episode lock."""
    path=OUT/'controller-reload-request.json'
    request=read(path,{})
    if not request.get('pending') or paused or done:return False
    with LOCK.open('a') as lock:
        if not acquire_update_lock(lock):return False
        # User instructions and the request may change while waiting for the
        # current episode to finish. Recheck before touching the controller.
        request=read(path,{})
        if not request.get('pending') or read(BATCH/'user-pause.json',{}).get('status')=='paused_by_user':return False
        if owner_alive(read(BATCH/'episode-slot-owner.json',{})) or native_learners():return False
        revisions=request.get('frontend_revisions',{})
        if revisions:
            if any(not str(k).isdigit() or not isinstance(v,str) or not re.fullmatch(r'_[A-Za-z0-9_]{1,80}',v) for k,v in revisions.items()):
                raise ValueError('Unsafe pending frontend revision')
            current=read(OUT/'frontend-revisions.json',{})
            current.update({str(int(k)):v for k,v in revisions.items()})
            save(OUT/'frontend-revisions.json',current)
        subprocess.run(['systemctl','--user','restart',CONTROLLER],check=True,timeout=20)
        request.update(pending=False,applied_utc=utc(),applied_without_an_active_episode=True)
        if revisions:request['applied_frontend_revisions']=revisions
        save(path,request)
        return True


def main():
    now = time.time()
    state = read(OUT / 'status.json', {})
    owner = read(BATCH / 'episode-slot-owner.json', {})
    paused = read(BATCH / 'user-pause.json', {}).get('status') == 'paused_by_user'
    controller = service_state(CONTROLLER)
    gui = service_state(GUI)
    owned = owner_alive(owner)
    learners = native_learners()
    plan = state.get('execution_plan', {})
    coverage = {int(key) for key,row in plan.items() if str(key).isdigit() and row.get('automatic',True)}
    verified_done,invalid_claims,completion_checks=strict_progress(state,EXPECTED)
    done = set(verified_done) == EXPECTED
    queue_age = age(state.get('updated_utc'), now)
    audit_age = age(read(OUT / 'audit-progress.json', {}).get('updated_utc'), now)
    activity_age = min([a for a in [queue_age, audit_age] if a is not None], default=None)
    job = state.get('active_optimization') or state.get('external_optimization') or {}
    root = Path(job['run']) if job.get('run') else None
    if root is None and owned and owner.get('run'):
        root = Path(owner['run'])
    if root is None and job.get('generation') is not None:
        root = BASE / f"runs/episode_{job['episode']:06d}_newton_human_match_20261023_g{job['generation']:03d}"
    metric = latest_metric(root / 'flash_chord_coherent_actor_transfer/metrics.jsonl') if root else {}
    metric_age = age(metric.get('utc'), now)
    ram = available_ram_bytes()
    disk = shutil.disk_usage(BASE).free
    alerts = []
    idle_problem=source_alignment_idle_problem(state,owned,paused)
    if idle_problem:alerts.append(idle_problem)
    if invalid_claims:
        alerts.append('Completion claims lack strict evidence for episodes: '+', '.join(map(str,invalid_claims)))
    if coverage != EXPECTED:
        alerts.append(f'Automatic execution plan does not contain exactly the {len(EXPECTED)} focused episodes.')
    if len(learners) > 1:
        alerts.append('More than one challenge policy learner is running; inspect GPU admission.')
    if controller == 'active' and activity_age is not None and activity_age > 180:
        alerts.append(f'Queue heartbeat is {int(activity_age)} seconds old.')
    if learners and metric_age is not None and metric_age > 900:
        alerts.append(f'Policy metrics have not advanced for {int(metric_age)} seconds.')
    if ram < 8 * 1024**3:
        alerts.append('Available RAM is below 8 GiB; new jobs already require 12 GiB at admission.')
    if disk < 30_000_000_000:
        alerts.append('Free disk space is below 30 GB; preserve completed artifacts and inspect storage.')
    if controller != 'active' and owned and not paused:
        alerts.append('Queue controller is unavailable; waiting for the existing episode process to exit before recovery.')
    history=state.get('optimization_history',[])
    source_cad=read(OUT/'source-cad-status.json',{}) if state.get('status') in {'source_cad_alignment','evaluation_frontend'} else {}
    required=state.get('required_episode')
    failures=[]
    for job in reversed(history):
        if job.get('episode')!=required:continue
        if not job.get('exit_code'):break
        failures.append(job)
    if len(failures)>=3 and not owned and not source_cad:
        alerts.append(f'Episode {required:06d} has {len(failures)} consecutive failed repair attempts; no policy learner is running.')
    validation=read(OUT/'pipeline_validation_20261029/status.json',{})
    public_task,public_problem=public_validation_health(validation,owned,paused)
    if state.get('status')=='public_pipeline_validation' and public_problem:
        alerts.append(public_problem)
    actions = []
    experimental=read(OUT/'experimental-training-status.json',{})
    replay=read(OUT/'submission-replay-status.json',{})
    if replay.get('status')=='blocked':
        alerts.append('Official policy replay is blocked; see '+str(replay.get('failure_log','submission-replay-status.json')))
    if experimental.get('status')=='pilot_blocked':
        detail=str(experimental.get('failure_summary') or experimental.get('latest_failure') or 'See pilot logs').strip().splitlines()[-1]
        alerts.append('Experimental policy preparation is blocked: '+detail)
    submission_complete=bool(state.get('submission_cohort_completed'))
    if apply_pending_update(paused,done or submission_complete):
        actions.append('Reloaded collision-aware queue checks between episodes; no learner was interrupted.')
    if recovery_decision(controller, paused, done or submission_complete, owned):
        subprocess.run(['systemctl', '--user', '--no-block', 'start', CONTROLLER], check=True, timeout=10)
        actions.append('Started unavailable queue controller; no existing episode process was alive.')
    if gui in {'inactive', 'failed'}:
        subprocess.run(['systemctl', '--user', '--no-block', 'start', GUI], check=True, timeout=10)
        actions.append('Started unavailable GUI service.')
    report = {
        'updated_utc': utc(), 'interval_seconds': 60,
        'status': 'paused_by_user' if paused else 'needs_attention' if alerts else 'healthy',
        'controller_service': controller, 'gui_service': gui,
        'queue_heartbeat_age_seconds': queue_age,
        'required_episode': state.get('required_episode'),
        'current_episode': owner.get('episode') if owned else state.get('current_episode'),
        'queue_order': state.get('serial_queue_order', []),
        'registered_episodes': sorted(coverage), 'registered_episode_count': len(coverage),
        'expected_episode_count':len(EXPECTED),
        'episodes_done': verified_done,
        'reported_episodes_done': state.get('episodes_done', []),
        'strict_completion_independently_rechecked':True,
        'required_episode_blockers':completion_checks.get(state.get('required_episode'),{}).get('blocking_reasons',[]),
        'native_learner_pids': learners, 'owned_episode_process_alive': owned,
        'run': str(root) if root else None,
        'environment_steps': metric.get('environment_steps'),
        'metrics_age_seconds': metric_age,
        'available_ram_gib': ram / 1024**3, 'free_disk_gb': disk / 1e9,
        'alerts': alerts, 'recovery_actions': actions,
        'exact_motion_before_advancing': state.get('require_verified_match_before_next') is True,
        'launches_episode_workers': False,
        'controller_update_pending': bool(read(OUT/'controller-reload-request.json',{}).get('pending')),
        'task_status': 'paused_by_user' if paused else 'submission_cohort_completed' if submission_complete else 'finished' if done else 'training' if learners else 'reconstructing_or_validating' if owned else 'waiting_to_retry_failed_episode' if failures else state.get('status'),
        'consecutive_current_episode_failures':len(failures),
        'source_cad':source_cad,
    }
    if source_cad:
        report['task_status']='source_cad_alignment' if owned else source_cad.get('status')
        report['active_stage']=source_cad.get('progress',{}).get('phase',source_cad.get('stage'))
    if state.get('status')=='public_pipeline_validation' and validation:
        report.update(task_status=public_task,active_stage=validation.get('phase'),
                      dataset_split='public',public_validation=validation,
                      public_blocking_failure=public_problem,
                      environment_steps=0 if not learners else report['environment_steps'])
    save(OUT / 'health-monitor.json', report)
    if alerts or actions:
        with (OUT / 'health-events.jsonl').open('a') as stream:
            stream.write(json.dumps(report) + '\n')
    print(json.dumps(report), flush=True)


if __name__ == '__main__':
    main()
