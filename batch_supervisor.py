"""Keep unfinished episode queues progressing after a worker exits; never mark missing videos complete."""
from pathlib import Path
import fcntl,hashlib,json,os,shutil,subprocess,sys,time
from batch_artifacts import complete
BASE=Path('/home/yu_eli/v2d-track3');B=BASE/'runs/evaluation_batch_20261015';MAIN=[3,6,8,10,15,19,20,25,28,32,33];TAIL=[35,36,14,16,30,37,38]
def free_lock(name):
 with (B/name).open('a') as lock:
  try:fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
  except BlockingIOError:return False
  return True
def finishers():
 result=set()
 for p in Path('/proc').glob('[0-9]*/cmdline'):
  try:
   argv=p.read_bytes().split(b'\0')
   if any(x.endswith(b'/batch_finish_existing.py') for x in argv):result.update(int(x) for x in argv if x.isdigit())
  except (OSError,ValueError):pass
 return result
def pending(episodes,skip):
 return [ep for ep in episodes if ep not in skip and not complete(BASE/'runs'/f'episode_{ep:06d}_newton_batch_20261015')]
def fingerprint(ep):
 r=BASE/'runs'/f'episode_{ep:06d}_newton_batch_20261015';paths=[r/'episode-config.json',BASE/'batch_native.py',BASE/'batch_review.py',BASE/'batch_export.py',BASE/'batch_generic_retarget.py',BASE/'batch_training_queue.py',BASE/'batch_tail_training.py',r/'scripts/view_policy_single_attempt.py'];h=hashlib.sha256()
 for p in paths:h.update(p.read_bytes())
 return h.hexdigest()
with (B/'supervisor.lock').open('a') as supervisor_lock:
 fcntl.flock(supervisor_lock,fcntl.LOCK_EX|fcntl.LOCK_NB);history=B/'supervisor-state.json';state=json.loads(history.read_text()) if history.exists() else {'launches':[],'attempts':{}}
 while True:
  if all(complete(BASE/'runs'/f'episode_{ep:06d}_newton_batch_20261015') for ep in MAIN+TAIL):print('All 18 validated episode artifact sets exist.',flush=True);break
  for script,lockname,episodes in [('batch_training_queue.py','training-queue.lock',MAIN),('batch_tail_training.py','tail-training-queue.lock',TAIL)]:
   if not free_lock(lockname):continue
   todo=[]
   for ep in pending(episodes,finishers()):
    signature=fingerprint(ep);key=str(ep);old=state['attempts'].get(key,{});count=old.get('count',0) if old.get('signature')==signature else 0
    if count>=3:continue
    r=BASE/'runs'/f'episode_{ep:06d}_newton_batch_20261015';log=r/'logs/newton-training.log'
    if log.exists() and 'Traceback' in log.read_text()[-6000:] and not (r/'flash_chord_coherent_actor_transfer/TRAINING_COMPLETED').exists():
     training_lock=r/'flash_chord_coherent_actor_transfer/training-process.lock'
     with training_lock.open('a') as attempt_lock:
      try:fcntl.flock(attempt_lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
      except BlockingIOError:continue
      backup=r/f'research/training-failure-{time.time_ns()}.log';shutil.copy2(log,backup);log.write_text(f'Previous failure archived at {backup.name}; waiting for a resumable training attempt.\n')
    todo.append(ep);state['attempts'][key]={'signature':signature,'count':count+1}
   if not todo:continue
   logfile=B/f'supervised-{Path(script).stem}-{time.time_ns()}.log'
   with logfile.open('w') as log:process=subprocess.Popen([sys.executable,'-u',str(BASE/script),*map(str,todo)],cwd=BASE,stdin=subprocess.DEVNULL,stdout=log,stderr=subprocess.STDOUT,start_new_session=True)
   entry={'pid':process.pid,'script':script,'episodes':todo,'log':logfile.name,'utc':time.strftime('%Y-%m-%dT%H:%M:%SZ',time.gmtime())};state['launches'].append(entry);print('Queue recovery launched',entry,flush=True)
   temporary=history.with_suffix('.tmp');temporary.write_text(json.dumps(state,indent=2));temporary.replace(history)
  time.sleep(30)
