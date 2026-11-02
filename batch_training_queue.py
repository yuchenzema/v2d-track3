"""Run reviewed episodes through actual FlashSAC training, checkpoint selection and export."""
from pathlib import Path
import json,sys,time,subprocess,concurrent.futures,traceback,fcntl
from batch_worker import context,BASE
from batch_review import monitor
from batch_export import export
from batch_generic_retarget import install
from batch_collision_reference import prepare as prepare_collision_reference
from batch_refine_contacts import refine as refine_contacts
from batch_reconstruction_queue import status
from batch_artifacts import complete

def reviewed_episodes(episodes):
 """Keep ready work moving while another reconstruction is being corrected."""
 pending=list(episodes)
 while pending:
  for ep in list(pending):
   r,c=context(ep);p=r/'pipeline-status.json';s=json.loads(p.read_text()) if p.exists() else {}
   if complete(r) or (s.get('reconstruction',{}).get('status')=='completed' and c.get('reconstruction_visual_reviewed',False)):
    pending.remove(ep);yield ep;break
  else:time.sleep(15)

if __name__=='__main__':
 lock=open(BASE/'runs/evaluation_batch_20261015/training-queue.lock','w');fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
 # Episode 3 has its own active reviewer. Its final rendering can overlap the
 # next learner after its training process has actually finished.
 r3,c3=context(3)
 while not (r3/'flash_chord_coherent_actor_transfer/TRAINING_COMPLETED').exists():time.sleep(15)
 def finish_episode3():
  while not (r3/'final-controller-selection.json').exists():time.sleep(15)
  if not complete(r3):export(3)
  status(r3,'pipeline','completed',report='result-report.json')
 exports=concurrent.futures.ThreadPoolExecutor(max_workers=1);episode3_export=exports.submit(finish_episode3)
 failed=[]
 for ep in reviewed_episodes(map(int,sys.argv[1:])):
  r,c=context(ep)
  if complete(r):continue
  try:
   status(r,'pipeline','running',worker='main training queue')
   while True:
    c=json.loads((r/'episode-config.json').read_text());p=r/'pipeline-status.json';s=json.loads(p.read_text()) if p.exists() else {};ready=s.get('reconstruction',{}).get('status')=='completed' and c.get('reconstruction_visual_reviewed',False)
    if ready:break
    time.sleep(15)
   if not (r/'coherent-input-checksums.json').exists():
    prepare_collision_reference(ep);refine_contacts(ep)
   install(r)
   status(r,'training','running')
   if not (r/'flash_chord_coherent_actor_transfer/TRAINING_COMPLETED').exists():
    with concurrent.futures.ThreadPoolExecutor(max_workers=2) as pool:
     training=pool.submit(subprocess.run,[sys.executable,str(BASE/'batch_native.py'),str(ep),'train'],check=True)
     reviews=pool.submit(monitor,ep)
     training.result();reviews.result()
   elif not (r/'final-controller-selection.json').exists():monitor(ep)
   status(r,'training','completed');status(r,'rollout','running');export(ep);status(r,'rollout','completed');status(r,'pipeline','completed',report='result-report.json')
  except Exception:
   status(r,'pipeline','failed',traceback=traceback.format_exc());print('FAILED',ep,traceback.format_exc(),flush=True)
   failed.append(ep)

 episode3_export.result();exports.shutdown()
 if failed:raise SystemExit(1)
