"""Transfer the next episode16 training slot to its validated stationary-hold reconstruction."""
import fcntl,hashlib,json,os,shutil,signal,threading,time,traceback
from pathlib import Path
from improvement_common import BASE,BATCH,save,utc,baseline,better
from batch_native import _docker
from fresh_improvement_run import run
from publish_checkpoint_preview import publish
OLD=BASE/'runs/episode_000016_newton_grasp_curriculum_20261020';R=BASE/'runs/episode_000016_newton_stationary_hold_20261020'
OUT=BATCH/'improvements_20261020_contacts';PHASE=OUT/'stationary-continuation-phase.json';PRIORITY=BATCH/'training-priority.json'
def state(phase,**extra):save(PHASE,{'episode':16,'phase':phase,'updated_utc':utc(),'run':str(R),'superseded_run':str(OLD),**extra})
def metadata(path):
 with path.open('rb') as f:n=int.from_bytes(f.read(8),'little');h=json.loads(f.read(n))
 assert path.stat().st_size==8+n+max(v['data_offsets'][1] for k,v in h.items() if k!='__metadata__')
 return {k.removeprefix('user.'):v for k,v in h['__metadata__'].items() if k.startswith('user.')}
def restore_priority():
 p=json.loads(PRIORITY.read_text());p['allowed_run_names']=list(dict.fromkeys(p.get('allowed_run_names',[])+p.pop('stationary_transfer_deferred_run_names',[])));p['updated_utc']=utc();save(PRIORITY,p)
def previews(stop):
 published=set();failures={}
 while not stop.is_set():
  try:
   p=R/'checkpoint-reviews.json';rows=json.loads(p.read_text()) if p.exists() else [];previous=baseline(16)
   for row in rows:
    step=row['environment_steps']
    if step in published or failures.get(step,0)>=2 or row['additional_environment_steps']<5_000_000 or not better(row,previous):continue
    try:publish(16,step,'stationary_hold_20261020');published.add(step)
    except Exception:failures[step]=failures.get(step,0)+1;print('Continuation preview attempt failed',step,traceback.format_exc(),flush=True)
  except Exception:print('Continuation preview scan retry',traceback.format_exc(),flush=True)
  stop.wait(20)
def phase_monitor(stop):
 restored=False
 while not stop.is_set():
  p=R/'flash_chord_coherent_actor_transfer/metrics.jsonl';metrics={}
  if p.exists():
   try:metrics=json.loads(p.read_text().splitlines()[-1])
   except (IndexError,ValueError):pass
  if metrics:
   if not restored:restore_priority();restored=True
   state('training',additional_environment_steps=metrics['additional_environment_steps'],latest_metric_utc=metrics['utc'],remaining_training_step_budget=35_008_000)
  stop.wait(15)
with (OUT/'stationary-continuation.lock').open('a') as lock:
 fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
 assert json.loads((R/'research/preparation-status.json').read_text())['status']=='prepared_and_configuration_validated'
 deadline=time.monotonic()+1200
 while True:
  choices=[]
  for p in (OLD/'flash_chord_coherent_actor_transfer').glob('state_*.safetensors'):
   step=int(p.stem.split('_')[-1]);a=p.with_name(p.name.replace('state_','policy_',1))
   if step<369_900_000 or not a.exists() or min(time.time()-p.stat().st_mtime,time.time()-a.stat().st_mtime)<5:continue
   try:
    m,n=metadata(p),metadata(a);assert m['checkpoint_set_id']==n['checkpoint_set_id'];assert int(m['environment_steps'])==step;choices.append((step,p,a))
   except (AssertionError,KeyError,ValueError):continue
  if choices:break
  state('waiting_for_saved_legacy_checkpoint',training_continues_in_previous_run=True)
  assert time.monotonic()<deadline,'Legacy20M checkpoint unavailable within20 minutes; previous learner remains untouched'
  time.sleep(10)
 step,learner,actor=max(choices);archive=R/'research/previous-run-final-checkpoint';archive.mkdir(exist_ok=True)
 for p in [learner,actor]:shutil.copy2(p,archive/p.name);assert hashlib.sha256(p.read_bytes()).hexdigest()==hashlib.sha256((archive/p.name).read_bytes()).hexdigest()
 priority=json.loads(PRIORITY.read_text());deferred=[n for n in priority.get('allowed_run_names',[]) if any(n.startswith(f'episode_{ep:06d}_') for ep in [19,30,33])]
 priority['allowed_run_names']=list(dict.fromkeys([n for n in priority.get('allowed_run_names',[]) if n not in deferred]+[R.name]));priority['stationary_transfer_deferred_run_names']=deferred;priority['updated_utc']=utc();save(PRIORITY,priority)
 save(OLD/'research/superseded-training.json',{'updated_utc':utc(),'superseded_by_run':str(R),'preserved_latest_checkpoint':step,
  'continuation_seed_checkpoint':365000192,'reason':'Replace spurious seated-spoon motion and clenched nonpinching fingers with the visually checked, collision-audited and gravity-tested reference.',
  'old_training_not_marked_complete':True,'previous_inputs_and_videos_unchanged':True})
 save(R/'research/improvement-status.json',{'episode':16,'status':'waiting','updated_utc':utc(),'run':str(R)})
 code="""import os,signal
from pathlib import Path
found=[]
for p in Path('/proc').glob('[0-9]*/cmdline'):
 try:a=p.read_bytes().split(b'\\0')
 except OSError:continue
 if len(a)>2 and a[1]==b'-u' and a[2]==b'/episode/scripts/train_coherent_actor_transfer.py':found.append(int(p.parent.name))
assert len(found)==1,found
os.kill(found[0],signal.SIGTERM)
print('Stopped only the superseded16 learner; all checkpoints retained',found[0])
"""
 try:
  print(_docker(['exec','v2d-grasp-episode16-newton','python','-c',code]).stdout,flush=True);state('waiting_for_legacy_worker_handoff',preserved_legacy_checkpoint=step)
  deadline=time.monotonic()+600
  while True:
   p=OUT/'episode_000016-fresh-result.json';old=json.loads(p.read_text()) if p.exists() else {}
   if old.get('status')=='failed':break
   assert time.monotonic()<deadline,'Legacy host did not finish its bounded attempt within10 minutes'
   time.sleep(10)
  stop=threading.Event();threads=[threading.Thread(target=previews,args=(stop,),daemon=True),threading.Thread(target=phase_monitor,args=(stop,),daemon=True)]
  for t in threads:t.start()
  result=None
  try:
   for attempt in range(2):
    try:result=run(16,'stationary_hold_20261020');break
    except Exception:print('Stationary continuation attempt failed',attempt+1,traceback.format_exc(),flush=True)
  finally:stop.set()
  assert result is not None,'Both stationary-hold continuation attempts failed'
  save(OUT/'episode_000016-stationary-fresh-result.json',result);state('completed',measured_improvement=result.get('measured_improvement',False),training_complete=True)
 finally:restore_priority()
