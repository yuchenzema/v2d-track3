"""Bounded recovery for the prioritized grasp-first retraining queue."""
from pathlib import Path
import fcntl
import hashlib
import json
import subprocess
import time
from improvement_common import BASE,save,utc
from grasp_curriculum_queue import OUT


if __name__=='__main__':
    OUT.mkdir(exist_ok=True)
    with (OUT/'supervisor.lock').open('a') as lock:
        fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
        while True:
            phase=OUT/'phase-status.json'
            if phase.exists() and json.loads(phase.read_text()).get('phase')=='completed':break
            path=OUT/'worker.json';worker=json.loads(path.read_text()) if path.exists() else {}
            proc=Path(f'/proc/{worker.get("pid",0)}/cmdline')
            alive=proc.exists() and str(BASE/'grasp_curriculum_queue.py').encode() in proc.read_bytes().split(b'\0')
            if not alive:
                signature=hashlib.sha256(b''.join((BASE/name).read_bytes() for name in
                    ['grasp_curriculum_queue.py','grasp_curriculum_training.py','fresh_improvement_run.py','batch_native.py'])).hexdigest()
                statepath=OUT/'supervisor-state.json';state=json.loads(statepath.read_text()) if statepath.exists() else {}
                attempts=state.get('attempts',0) if state.get('signature')==signature else 0
                if attempts>=3:
                    save(OUT/'supervisor-attention.json',{'updated_utc':utc(),'reason':'Grasp-first queue exited on three recovery attempts with the same code'});break
                log=OUT/f'recovery-{time.time_ns()}.log'
                with log.open('w') as stream:
                    p=subprocess.Popen(['/usr/bin/python3','-u',str(BASE/'grasp_curriculum_queue.py')],cwd=BASE,
                         stdin=subprocess.DEVNULL,stdout=stream,stderr=subprocess.STDOUT,start_new_session=True)
                worker.update(pid=p.pid,log=str(log),updated_utc=utc());save(path,worker)
                save(statepath,{'signature':signature,'attempts':attempts+1,'updated_utc':utc()})
            time.sleep(30)
