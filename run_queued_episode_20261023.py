"""Execute one durable queue stage without sharing a reference with another worker."""
import argparse
import fcntl
import traceback
import time
import rerun_four_episodes_20261023 as work
from improvement_common import save, utc

p=argparse.ArgumentParser();p.add_argument('episode',type=int,choices=work.EPISODES)
p.add_argument('stage',choices=['reconstruct','train_rollout']);a=p.parse_args()
r,_=work.prepare(a.episode)
with (r/'research/queue-stage.lock').open('a') as lock:
    fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
    try:
        if a.stage=='reconstruct':work.rebuild(a.episode)
        else:
            native=work.make_container(r,a.episode,'newton')
            # Docker learners may outlive their host client. Adopt an existing
            # learner instead of launching a duplicate after service recovery.
            while not (r/'flash_chord_coherent_actor_transfer/TRAINING_COMPLETED').exists():
                top=work._docker(['top',native,'-eo','pid,args']).stdout
                if not any('/episode/scripts/train_coherent_actor_transfer.py' in line for line in top.splitlines()):break
                work.status(a.episode,'Monitoring the existing learner after queue recovery')
                time.sleep(15)
            work.train_and_export(a.episode)
    except Exception:
        error=traceback.format_exc()
        save(r/'research/queue-stage-error.json',{'updated_utc':utc(),'stage':a.stage,'error':error})
        work.status(a.episode,'Stage failed; automatic queue recovery pending',state='failed',error=error)
        raise
