"""Restart an exited improvement orchestrator with bounded, recorded retries."""
import fcntl
import hashlib
import json
from pathlib import Path
import subprocess
import time
from improvement_common import BASE,OUT,save,utc

SOURCES=['batch_improvement_queue.py','improvement_common.py','fresh_improvement_run.py',
         'fresh_improvement_training.py','residual_gain_controller.py','batch_review.py',
         'batch_export.py','render_improved_controller.py','object_motion_quality.py']


def active(pid):
    try:
        return str(BASE/'batch_improvement_queue.py').encode() in Path(f'/proc/{pid}/cmdline').read_bytes().split(b'\0')
    except OSError:return False


if __name__=='__main__':
    OUT.mkdir(exist_ok=True)
    with (OUT/'supervisor.lock').open('a') as lock:
        fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
        while True:
            status=OUT/'phase-status.json'
            phase=json.loads(status.read_text()).get('phase') if status.exists() else None
            if phase in ['completed','completed_with_experiment_failures']:break
            worker_path=OUT/'worker.json'
            worker=json.loads(worker_path.read_text()) if worker_path.exists() else {}
            if not active(worker.get('pid',0)):
                signature=hashlib.sha256(b''.join((BASE/name).read_bytes() for name in SOURCES)).hexdigest()
                old=OUT/'supervisor-state.json';state=json.loads(old.read_text()) if old.exists() else {}
                attempts=state.get('attempts',0) if state.get('signature')==signature else 0
                if attempts>=3:
                    save(OUT/'supervisor-attention.json',{'updated_utc':utc(),'reason':'Improvement orchestrator exited on three recovery attempts with the same code','phase':phase,'worker':worker})
                    break
                log=OUT/f'recovery-{time.time_ns()}.log'
                with log.open('w') as stream:
                    p=subprocess.Popen(['/usr/bin/python3','-u',str(BASE/'batch_improvement_queue.py')],cwd=BASE,
                                       stdin=subprocess.DEVNULL,stdout=stream,stderr=subprocess.STDOUT,start_new_session=True)
                worker.update(pid=p.pid,script=str(BASE/'batch_improvement_queue.py'),log=str(log),last_recovery_utc=utc())
                save(worker_path,worker);save(old,{'signature':signature,'attempts':attempts+1,'last_worker_pid':p.pid,'updated_utc':utc()})
                print('Recovered improvement orchestrator',p.pid,'attempt',attempts+1,flush=True)
            time.sleep(30)
