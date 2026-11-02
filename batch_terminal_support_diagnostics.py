"""Check final support poses independently of the learner; never export these as policy videos."""
import fcntl,json,shutil,sys
from batch_native import containers,docker
from batch_worker import context,BASE
for ep in map(int,sys.argv[1:]):
 r,c=context(ep)
 with (r/'research/collision-reference.lock').open('a') as lock:
  fcntl.flock(lock,fcntl.LOCK_EX);newton,_=containers(r,ep);shutil.copy2(BASE/'diagnose_terminal_support.py',r/'scripts/diagnose_terminal_support.py')
  docker(['exec',newton,'python','-u','/episode/scripts/diagnose_terminal_support.py'],r/'logs/terminal-support-diagnostic.log');print(ep,json.loads((r/'evaluation/terminal-support-diagnostic.json').read_text())['objects'],flush=True)
