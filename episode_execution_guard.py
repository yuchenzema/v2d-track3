"""One host-wide episode slot, including reconstruction, training and export."""
from contextlib import contextmanager
import fcntl
import hashlib
import json
import os
from pathlib import Path
import time
from improvement_common import BATCH,save,utc

LOCK=Path('/tmp/v2d-track3-single-episode.lock')
OWNER=BATCH/'episode-slot-owner.json'

def process_signature(pid):
    try:
        command=Path(f'/proc/{int(pid)}/cmdline').read_bytes()
        return hashlib.sha256(command).hexdigest() if command else None
    except (OSError,ValueError,TypeError):return None

def owner_alive(owner):
    for process in [owner]+owner.get('protected_children',[]):
        signature=process_signature(process.get('pid'))
        if signature and signature==process.get('process_signature'):return True
    return False

def register_episode_child(pid):
    """Retain the episode reservation if a Docker learner outlives its client."""
    try:owner=json.loads(OWNER.read_text())
    except (OSError,ValueError):return
    if owner.get('pid')!=os.getpid():return
    signature=process_signature(pid)
    if signature:
        owner.setdefault('protected_children',[]).append({'pid':pid,'process_signature':signature})
        save(OWNER,owner)

def available_ram_bytes():
    for line in Path('/proc/meminfo').read_text().splitlines():
        if line.startswith('MemAvailable:'):return int(line.split()[1])*1024
    return 0

@contextmanager
def episode_slot(root,episode):
    with LOCK.open('a') as lock:
        while True:
            fcntl.flock(lock,fcntl.LOCK_EX)
            try:owner=json.loads(OWNER.read_text())
            except (OSError,ValueError):owner={}
            if not owner_alive(owner) and available_ram_bytes()>=12*1024**3:break
            fcntl.flock(lock,fcntl.LOCK_UN);time.sleep(10)
        signature=process_signature(os.getpid())
        save(OWNER,{'episode':episode,'run':str(root),'pid':os.getpid(),'process_signature':signature,'acquired_utc':utc()})
        try:yield
        finally:
            try:owner=json.loads(OWNER.read_text())
            except (OSError,ValueError):owner={}
            if any(owner_alive(child) for child in owner.get('protected_children',[])):
                owner['status']='waiting_for_owned_learner_to_exit';save(OWNER,owner)
            else:save(OWNER,{'episode':None,'pid':None,'released_utc':utc()})
