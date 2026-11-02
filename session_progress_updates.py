"""Queue user-requested periodic status reports in the existing Codex session."""
from datetime import datetime, timezone
import fcntl
import json
from pathlib import Path
import subprocess
import time
import uuid

from improvement_common import BASE, BATCH, save, utc

OUT=BATCH/'focused_episodes_20261028'
CONFIG=OUT/'session-updates.json'
TIMER='v2d-track3-session-updates.timer'
CODEX=Path('/home/yu_eli/.local/bin/codex')


def read(path, default):
    try:return json.loads(path.read_text())
    except (OSError,ValueError):return default


def delivery_due(config, scope, now):
    if not config.get('enabled'):return False,'disabled'
    if scope.get('status')!='active' or scope.get('episodes')!=config.get('episodes'):
        return False,'focus_scope_changed_or_stopped'
    if config.get('final_report_queued'):return False,'all_focused_episodes_completed'
    if now-float(config.get('last_queued_timestamp',0))<600:return False,'not_due'
    return True,'due'


def main():
    with (OUT/'session-update-delivery.lock').open('a') as lock:
        try:fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
        except BlockingIOError:return
        config=read(CONFIG,{})
        due,reason=delivery_due(config,read(BATCH/'focus-scope.json',{}),time.time())
        if not due:
            print(reason,flush=True)
            if reason in {'focus_scope_changed_or_stopped','all_focused_episodes_completed'}:
                config.update(enabled=False,stopped_reason=reason,updated_utc=utc());save(CONFIG,config)
                subprocess.run(['systemctl','--user','disable','--now',TIMER],check=True,timeout=10)
            return
        thread=str(uuid.UUID(config['thread_id']))
        health=read(OUT/'health-monitor.json',{})
        complete=(health.get('strict_completion_independently_rechecked') is True
                  and set(health.get('episodes_done',[]))==set(config['episodes']))
        prompt=(
            'Scheduled 10-minute progress update requested by the user for this same session. '
            f'Read fresh focused queue status and logs under {OUT}. '
            f"First read pipeline_validation_20261029/status.json: report public video validation separately from the {len(config['episodes'])} evaluation episodes. "
            'Do not report public identity-control scores as reconstruction or policy scores. '
            'Also read source-cad-status.json and the active run research/cad-progress.json: '
            'the user requested CAD geometry and per-frame source-video overlay alignment first. '
            'Distinguish CAD overlays inspected from strict episode completion; an inspected overlay is not a passed episode. '
            'Report the active episode and stage, actual training steps if any, latest blocking failure, '
            f"and strict completion count out of {len(config['episodes'])}; keep it brief. Focus on episodes "
            + ','.join(map(str,config['episodes'])) + '; keep the GUI current. Preserve the serial worker and strict '
            'motion/contact checks. Do not create duplicate workers or a new task. '
            'Honor later user stop/pause/scope instructions. If no policy is training, say so.'
            ' Continue the already authorized full-pipeline improvement work after reporting: '
            'if a concrete blocker can be repaired, diagnose and fix it, verify the change, and resume the existing serial controller. '
            'Do not leave the queue idle merely because the automatic retry count was exhausted, and never weaken quality gates to advance.'
        )
        if complete:prompt+=' All selected episodes are reported strictly complete: verify the evidence and give a final report.'
        result=subprocess.run([str(CODEX),'queue','--thread',thread,'--message',prompt],
                              cwd=BASE,capture_output=True,text=True,timeout=35)
        now=time.time()
        if result.returncode:
            config.update(updated_utc=utc(),last_delivery_error=(result.stderr or result.stdout)[-1500:])
            save(CONFIG,config)
            raise RuntimeError('Could not queue the scheduled session update; see session-updates.json')
        config.update(updated_utc=utc(),last_queued_utc=utc(),last_queued_timestamp=now,
                      next_due_utc=datetime.fromtimestamp(now+600,timezone.utc).isoformat(),
                      queued_reports=int(config.get('queued_reports',0))+1,
                      last_queue_receipt=result.stdout.strip()[:1000],last_delivery_error=None)
        if complete:config['final_report_queued']=True
        save(CONFIG,config)
        print(json.dumps({'queued':True,'thread_id':thread,'next_due_utc':config['next_due_utc']}),flush=True)


if __name__=='__main__':main()
