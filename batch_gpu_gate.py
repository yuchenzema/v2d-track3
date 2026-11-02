"""Serialize large vision jobs and small native checks separately around one learner."""
from contextlib import contextmanager
from pathlib import Path
import fcntl,subprocess,time
from episode_execution_guard import available_ram_bytes
@contextmanager
def auxiliary_gpu(kind='vision',minimum_free_mb=3500):
 name='/tmp/v2d-track3-native-gpu-v2.lock' if kind=='native' else '/tmp/v2d-track3-auxiliary-gpu.lock'
 with Path(name).open('a') as lock:
  while True:
   fcntl.flock(lock,fcntl.LOCK_EX)
   result=subprocess.run(['nvidia-smi','--query-gpu=memory.free','--format=csv,noheader,nounits'],capture_output=True,text=True,check=True)
   if int(result.stdout.strip().splitlines()[0])>=minimum_free_mb+2000 and available_ram_bytes()>=12*1024**3:break
   # A larger pending render must not monopolize admission while smaller
   # checkpoint checks still fit in the remaining shared GPU memory.
   fcntl.flock(lock,fcntl.LOCK_UN);time.sleep(10)
  yield
