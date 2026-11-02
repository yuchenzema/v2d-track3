"""Local evaluation-only orchestration and terminal reports; no model calls.

The existing systemd controller owns all episode work. This process only
configures that controller, maintains its existing services, and reads evidence.
"""
import argparse
import csv
from datetime import datetime, timezone
import fcntl
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
import time

from episode_completion_gate import completion

BASE = Path(__file__).resolve().parent
OUT = BASE/'runs/evaluation_batch_20261015/focused_episodes_20261028'
EPISODES = [1,3,5,6,8,10,14,15,16,19,20,25,28,30,32,33,35,36,37,38]
CONTROLLER = 'v2d-track3-focused.service'
GUI = 'v2d-track3-gui.service'
HEALTH = 'v2d-track3-focused-health.timer'
MODEL_TIMER = 'v2d-track3-session-updates.timer'


def now():
    return datetime.now(timezone.utc).strftime('%Y-%m-%dT%H:%M:%SZ')


def read(path):
    try:
        return json.loads(Path(path).read_text())
    except (OSError, ValueError):
        return {}


def write(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name+f'.{os.getpid()}.tmp')
    temporary.write_text(json.dumps(value, indent=2, ensure_ascii=False)+'\n')
    os.replace(temporary, path)


def systemctl(*args, check=True):
    return subprocess.run(['systemctl', '--user', *args], check=check,
                          capture_output=True, text=True, timeout=40)


def recorded_steps(run):
    try:
        metrics=Path(run)/'flash_chord_coherent_actor_transfer/metrics.jsonl'
        with metrics.open('rb') as stream:
            stream.seek(max(0,metrics.stat().st_size-65536))
            lines=stream.read().decode(errors='replace').splitlines()
        for line in reversed(lines):
            try:value=json.loads(line)
            except ValueError:continue
            if 'environment_steps' in value:return value['environment_steps']
    except (OSError,TypeError):pass
    return None


def configure(out):
    """Change scope, never source/physics thresholds or completion evidence."""
    policy = read(out/'validated-pipeline-policy.json')
    if not policy.get('enabled'):
        raise RuntimeError('Existing validated pipeline policy is missing/disabled.')
    if policy.get('episodes') != EPISODES:
        raise RuntimeError('Existing policy does not contain the requested 20 episodes.')
    policy.update(public_validation_required=False, execution_scope='evaluation_only',
                  updated_utc=now())
    write(out/'validated-pipeline-policy.json', policy)
    source = read(out/'source-first-policy.json')
    source.update(enabled=True, episodes=EPISODES,
                  requires_cad_overlap_before_training=True,
                  preserve_strict_motion_and_contact_gates=True, updated_utc=now())
    write(out/'source-first-policy.json', source)
    updates = read(out/'session-updates.json')
    updates.update(enabled=False, stopped_reason='Local terminal reports; no model calls',
                   updated_utc=now())
    write(out/'session-updates.json', updates)


def start_services(out, resume=False):
    configure(out)
    systemctl('disable', '--now', MODEL_TIMER)
    pause = read(out.parent/'user-pause.json')
    if pause.get('status') == 'paused_by_user':
        if not resume:
            raise RuntimeError('User pause remains active. Use ./run_evaluation20.sh resume explicitly.')
        pause.update(status='resumed_by_user', updated_utc=now(),
                     reason='Explicit terminal resume', resumed_episodes=EPISODES)
        write(out.parent/'user-pause.json', pause)
    # systemctl start is idempotent: attach to the same running controller.
    systemctl('start', GUI, HEALTH, CONTROLLER)
    systemctl('start', 'v2d-track3-focused-health.service')


def snapshot(out):
    public = read(out/'pipeline_validation_20261029/status.json')
    state = read(out/'status.json')
    experimental=read(out/'experimental-training-status.json')
    cad = read(out/'source-cad-status.json')
    health = read(out/'health-monitor.json')
    source_active = state.get('status') in {'source_cad_alignment', 'evaluation_frontend'}
    active = (cad.get('active_episode') if source_active else
              state.get('current_episode'))
    job = state.get('active_optimization') or {}
    run = (cad.get('run') if source_active else job.get('run'))
    if not run and (str(state.get('status','')).startswith('experimental_policy_') or state.get('status') in {'pilot_blocked','pilot_finished'}):
        run=experimental.get('run')
        if run and active is None:active=read(Path(run)/'episode-config.json').get('episode')
    progress = read(Path(run)/'research/cad-progress.json') if run else {}
    if job.get('submission_replay'):progress={}
    # A cached CAD progress file must never label a different frontend stage.
    stage = cad.get('stage') if source_active else job.get('action') or state.get('phase')
    if job.get('experimental_only'):
        stage=state.get('phase') or stage
    if stage and stage.startswith('fresh_video_frontend_'):
        progress = {}
    checks = state.get('episodes') or {}
    done = [ep for ep in EPISODES if completion(checks.get(str(ep), {}))['episode_done']]
    results = {ep: cad.get('results', {}).get(str(ep), {}) for ep in EPISODES}
    learners = []
    owner=read(out.parent/'episode-slot-owner.json')
    owned={}
    if run and owner.get('run')==str(run):
        owned={p.get('pid'):p.get('process_signature') for p in owner.get('protected_children',[])}
    for pid in set(health.get('native_learner_pids', []))|set(owned):
        try:
            args = Path(f'/proc/{int(pid)}/cmdline').read_bytes()
            scoped=(pid in owned and hashlib.sha256(args).hexdigest()==owned[pid]) or (run and str(run).encode() in args)
            if scoped and b'train_coherent_actor_transfer.py' in args:
                learners.append(int(pid))
        except (OSError, ValueError, TypeError):
            pass
    failure = cad.get('latest_failure') if source_active else None
    if experimental.get('status')=='pilot_blocked':
        trained=bool(run and (Path(run)/'flash_chord_coherent_actor_transfer/TRAINING_COMPLETED').exists())
        prefix='訓練後驗證／匯出失敗：' if trained else '實驗訓練前流程失敗：'
        failure=prefix+str(experimental.get('failure_summary') or experimental.get('latest_failure') or 'See experimental-training-status.json')
        if trained and not source_active:stage=failure
    failed = sum(r.get('local_overlap_failed_object_frames', 0) for r in results.values())
    unobservable = sum(r.get('unobservable_object_frames', 0) for r in results.values())
    if not failure and failed and not job.get('experimental_only'):
        failure = f'來源 CAD 對齊仍有 {failed} 個失敗物件幀；不可觀測物件幀 {unobservable}。'
    if state.get('status') == 'source_alignment_repairs_required':
        failure = ('現有修正策略已用完；需要新的來源標註或演算法修正。'
                   '不會重設次數或降低門檻。 '+(failure or ''))
    steps=health.get('environment_steps') if learners else 0
    if run:
        measured=recorded_steps(run)
        if measured is not None:steps=measured
    history={}
    for ep,attempt in experimental.get('attempts',{}).items():
        count=recorded_steps(attempt.get('run'))
        if count:history[ep]=count
    if experimental.get('run'):
        cfg=read(Path(experimental['run'])/'episode-config.json');count=recorded_steps(experimental['run'])
        if count and cfg.get('episode') in EPISODES:history[str(cfg['episode'])]=count
    return dict(updated_utc=now(), scope='evaluation_only', active_episode=active,
                stage=stage, state=state.get('status'), run=run, cad_progress=progress,
                inspected=sum(bool(r) for r in results.values()),
                source_locally_passed=sum(bool(r) and r.get('all_source_frames_checked') is True
                    and r.get('local_overlap_failed_object_frames') == 0
                    and r.get('unobservable_object_frames') == 0 for r in results.values()),
                strict_completed=done, strict_completion_count=len(done),
                learner_pids=sorted(learners), training_steps=steps,
                recorded_training_steps_by_episode=history,
                blocking_failure=failure, health=health.get('status'),
                public_status=public.get('status'), quality_gates_unchanged=True)


def export_failures(out):
    """Bounded local diagnostics; never alter worker-owned reports or masks."""
    cad = read(out/'source-cad-status.json')
    destination = out/'terminal-diagnostics'
    destination.mkdir(exist_ok=True)
    with (destination/'failed-frames.csv').open('w', newline='') as stream:
        fields = ['episode','frame','object','iou','centroid_error_px',
                  'boundary_error_px','sphere_error_px','visibility','local_overlap_pass']
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        for ep in EPISODES:
            report = cad.get('results', {}).get(str(ep), {})
            path = report.get('frames_report')
            if not path:
                continue
            rows = read(path)
            if not isinstance(rows, list):
                continue
            for row in rows:
                if not row.get('local_overlap_pass'):
                    writer.writerow({'episode':ep, **{k:row.get(k) for k in fields if k != 'episode'}})
    return destination/'failed-frames.csv'


def render(report):
    p = report['cad_progress']
    progress = (f"；CAD {p.get('frame')}/{p.get('total_frames')} 幀，"
                f"{p.get('object_frames_processed')}/{p.get('total_object_frames')} 物件幀") if p else ''
    return '\n'.join([
        f"[{report['updated_utc']}] Evaluation 20 本機進度（不呼叫模型）",
        f"目前 episode={report['active_episode']}；階段={report['stage'] or report['state']}{progress}",
        f"CAD 已審核 {report['inspected']}/20；來源局部檢查通過 {report['source_locally_passed']}/20；嚴格完成 {report['strict_completion_count']}/20",
        (f"Policy learner 已啟動，steps={report['training_steps'] if report['training_steps'] is not None else '初始化／編譯中，等待首筆回報'}，PID={report['learner_pids']}" if report['learner_pids']
         else f"目前沒有 policy 在訓練；此 run 已記錄訓練 steps={report['training_steps'] or 0}。"),
        '已記錄的實驗訓練：'+(', '.join(f"episode {ep}={steps:,} steps" for ep,steps in report.get('recorded_training_steps_by_episode',{}).items()) or '尚無；品質驗收另計。'),
        f"阻塞：{str(report['blocking_failure'] or '目前沒有新的執行錯誤。')[-1600:]}",
        f"GUI http://localhost:8768；健康狀態={report['health']}；public 已排除。",
    ])


def main():
    loaded_version=tuple((BASE/name).stat().st_mtime_ns for name in ['evaluation20_terminal.py','episode_completion_gate.py'])
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('command', nargs='?', default='run', choices=['run','status','resume','pause'])
    parser.add_argument('--interval', type=int, default=600, help='Terminal report interval, seconds (default 600).')
    args = parser.parse_args()
    if args.interval < 1:
        parser.error('--interval must be positive')
    OUT.mkdir(parents=True, exist_ok=True)
    if args.command == 'status':
        print(render(snapshot(OUT)), flush=True)
        return
    if args.command == 'pause':
        pause = read(OUT.parent/'user-pause.json')
        pause.update(status='paused_by_user', updated_utc=now(), reason='Explicit terminal pause')
        write(OUT.parent/'user-pause.json', pause)
        print('已要求在目前單集結束後暫停；不再派送下一集。', flush=True)
        return
    with (OUT/'terminal-monitor.lock').open('a') as lock:
        try:
            fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError:
            raise SystemExit('另一個 terminal monitor 已在執行；可使用 status 查看。')
        start_services(OUT, resume=args.command == 'resume')
        print('沿用既有序列 controller；GUI 每 5 秒更新；terminal 每 '
              f'{args.interval} 秒報告。Ctrl+C 只結束監看，背景工作繼續。', flush=True)
        next_report = 0
        next_services = time.monotonic()+60
        try:
            while True:
                current_version=tuple((BASE/name).stat().st_mtime_ns for name in ['evaluation20_terminal.py','episode_completion_gate.py'])
                if current_version!=loaded_version:
                    print('監看程式已更新，重新載入；背景 controller 繼續。',flush=True)
                    os.execv(sys.executable,[sys.executable,'-u',str(BASE/'evaluation20_terminal.py'),*sys.argv[1:]])
                report = snapshot(OUT)
                write(OUT/'terminal-progress.json', report)
                current = time.monotonic()
                if current >= next_report or report['strict_completion_count'] == 20:
                    message = render(report)
                    print(message, flush=True)
                    with (OUT/'terminal-progress.log').open('a') as stream:
                        stream.write(message+'\n\n')
                    export_failures(OUT)
                    next_report = current+args.interval
                if report['strict_completion_count'] == 20:
                    print('20/20 嚴格完成證據已通過檢查。', flush=True)
                    return
                if current >= next_services:
                    paused = read(OUT.parent/'user-pause.json').get('status') == 'paused_by_user'
                    if not paused:
                        systemctl('start', GUI, HEALTH, CONTROLLER)
                    next_services = current+60
                time.sleep(min(15, args.interval))
        except KeyboardInterrupt:
            print('\nterminal 監看已結束；背景序列 controller 繼續執行。', flush=True)


if __name__ == '__main__':
    main()
