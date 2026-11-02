"""Run a second reviewed episode queue alongside the main learner when memory permits."""
from pathlib import Path
import json,sys,time,fcntl,subprocess,concurrent.futures,traceback,os
from batch_worker import context,BASE,PY
from batch_generic_retarget import install
from batch_collision_reference import prepare
from batch_refine_contacts import refine
from batch_review import monitor
from batch_export import export
from batch_reconstruction_queue import status
from batch_artifacts import complete

def free_gpu_mb():return int(subprocess.check_output(['nvidia-smi','--query-gpu=memory.free','--format=csv,noheader,nounits'],text=True).splitlines()[0])
def owns_training_process(r):
 path=r/'flash_chord_coherent_actor_transfer/training-process.lock'
 if not path.parent.exists():return False
 with path.open('a') as training_lock:
  try:fcntl.flock(training_lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
  except BlockingIOError:return True
 return False
def other_exports_pending(ep):
 for r in (BASE/'runs').glob('episode_*_newton_batch_20261015'):
  if r==context(ep)[0]:continue
  if (r/'flash_chord_coherent_actor_transfer/TRAINING_COMPLETED').exists() and not complete(r):return True
 return False
if __name__=='__main__':
 with (BASE/'runs/evaluation_batch_20261015/tail-training-queue.lock').open('a') as lock:
  fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
  for ep in map(int,sys.argv[1:]):
   assert ep in [14,16,28,30,35,36,37,38];r,c=context(ep)
   if complete(r):continue
   while True:
    c=json.loads((r/'episode-config.json').read_text());p=r/'pipeline-status.json';s=json.loads(p.read_text()) if p.exists() else {}
    if s.get('reconstruction',{}).get('status')=='completed' and c.get('reconstruction_visual_reviewed'):break
    time.sleep(15)
   try:
    status(r,'pipeline','running',worker='tail training queue')
    if not (r/'coherent-input-checksums.json').exists():prepare(ep);refine(ep)
    install(r)
    subprocess.run([PY,str(BASE/'assess_physical_reference.py'),'--run',str(r),'--reference',json.loads((r/'episode-config.json').read_text()).get('training_reference','track3_coherent_variant'),'--overlay'],check=True)
    # The native admission gate accounts for object count and permits at most
    # two learners. Reattach immediately to an already running own learner.
    # A failed or unattended export must not stall unrelated ready training.
    # The native admission gate and auxiliary GPU lock enforce actual memory
    # requirements for both training and rendering.
    while not (r/'flash_chord_coherent_actor_transfer/TRAINING_COMPLETED').exists() and not owns_training_process(r) and (free_gpu_mb()<7000 or os.statvfs(BASE).f_bavail*os.statvfs(BASE).f_frsize<2_000_000_000):time.sleep(15)
    status(r,'training','running',worker='independent tail queue')
    if not (r/'flash_chord_coherent_actor_transfer/TRAINING_COMPLETED').exists():
     with concurrent.futures.ThreadPoolExecutor(max_workers=2) as pool:
      trainer=pool.submit(subprocess.run,[sys.executable,str(BASE/'batch_native.py'),str(ep),'train'],check=True);reviewer=pool.submit(monitor,ep);trainer.result();reviewer.result()
    elif not (r/'final-controller-selection.json').exists():monitor(ep)
    status(r,'training','completed');status(r,'rollout','running');export(ep);status(r,'rollout','completed');status(r,'pipeline','completed',report='result-report.json')
   except Exception:
    status(r,'pipeline','failed',traceback=traceback.format_exc());print('Tail episode FAILED',ep,traceback.format_exc(),flush=True);raise
