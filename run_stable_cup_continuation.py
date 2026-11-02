"""Give the validated cup recovery the next saved training slot, preserving episode30."""
import fcntl,json,hashlib,threading,time,traceback
from improvement_common import BASE,BATCH,save,utc,baseline,better
from batch_native import _docker,active_learners
from fresh_improvement_run import run
from publish_checkpoint_preview import publish
R=BASE/'runs/episode_000010_newton_stable_contact_20261020';OLD=BASE/'runs/episode_000010_newton_grasp_curriculum_20261020'
PAUSE=BASE/'runs/episode_000030_newton_grasp_curriculum_20261020';OUT=BATCH/'improvements_20261020_contacts'
P=BATCH/'training-priority.json';PHASE=OUT/'stable-cup-continuation-phase.json'
def state(phase,**extra):save(PHASE,{'episode':10,'phase':phase,'run':str(R),'updated_utc':utc(),**extra})
def restore_priority():
 a=json.loads(P.read_text());a['allowed_run_names']=list(dict.fromkeys(a.get('allowed_run_names',[])+a.pop('stable_cup_deferred_run_names',[])));a['updated_utc']=utc();save(P,a)
def monitor_phase(stop):
 while not stop.is_set():
  p=R/'flash_chord_coherent_actor_transfer/metrics.jsonl'
  if p.exists():
   try:
    a=json.loads(p.read_text().splitlines()[-1]);restore_priority()
    state('training',additional_environment_steps=a['additional_environment_steps'],latest_metric_utc=a['utc'],remaining_training_step_budget=40_008_192,
     numerical_failure_resets=a.get('numerical_guard/failure_resets_total',0))
   except (ValueError,IndexError):pass
  stop.wait(15)
def previews(stop):
 done=set();errors={}
 while not stop.is_set():
  if (R/'flash_chord_coherent_actor_transfer/TRAINING_COMPLETED').exists():return
  p=R/'checkpoint-reviews.json'
  if p.exists():
   for row in json.loads(p.read_text()):
    step=row['environment_steps']
    if step in done or errors.get(step,0)>=2 or not better(row,baseline(10)):continue
    try:publish(10,step,'stable_contact_20261020');done.add(step)
    except Exception:errors[step]=errors.get(step,0)+1;print('Cup checkpoint preview retry',step,traceback.format_exc(),flush=True)
  stop.wait(20)
with (OUT/'stable-cup-continuation.lock').open('a') as lock:
 fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
 assert json.loads((R/'research/preparation-status.json').read_text())['status']=='prepared_and_configuration_validated'
 save(R/'research/improvement-status.json',{'episode':10,'status':'waiting','updated_utc':utc(),'run':str(R)})
 priority=json.loads(P.read_text());priority['allowed_run_names']=list(dict.fromkeys(priority.get('allowed_run_names',[])+[R.name]));save(P,priority)
 target=None;deadline=time.monotonic()+3600
 while True:
  top=_docker(['top','v2d-grasp-episode30-newton','-eo','pid,args']).stdout
  running='/episode/scripts/train_coherent_actor_transfer.py' in top
  if not running and len(active_learners())<3:break
  for path in (PAUSE/'flash_chord_coherent_actor_transfer').glob('state_*.safetensors'):
   step=int(path.stem.split('_')[-1]);actor=path.with_name(path.name.replace('state_','policy_',1))
   if step<34_900_000 or not actor.exists() or min(time.time()-path.stat().st_mtime,time.time()-actor.stat().st_mtime)<5:continue
   metas=[]
   for file in [path,actor]:
    with file.open('rb') as f:n=int.from_bytes(f.read(8),'little');header=json.loads(f.read(n))
    assert file.stat().st_size==8+n+max(x['data_offsets'][1] for k,x in header.items() if k!='__metadata__')
    metas.append(header['__metadata__'])
   assert metas[0]['user.checkpoint_set_id']==metas[1]['user.checkpoint_set_id'];target=step;break
  if target:break
  state('waiting_for_saved_training_slot',yielding_episode=30,required_saved_environment_steps=35_000_000)
  assert time.monotonic()<deadline,'Next saved training slot did not become available within1 hour'
  time.sleep(15)
 priority=json.loads(P.read_text());deferred=[n for n in priority.get('allowed_run_names',[]) if any(n.startswith(f'episode_{ep:06d}_') for ep in [19,30,33])]
 priority['allowed_run_names']=[n for n in priority.get('allowed_run_names',[]) if n not in deferred];priority['stable_cup_deferred_run_names']=deferred;save(P,priority)
 save(OLD/'research/superseded-training.json',{'updated_utc':utc(),'superseded_by_run':str(R),'reason':'Nonfinite actor loss; preserve prior artifacts and recover with finer physics, float32 learning and fresh finite learner state.','old_training_not_marked_complete':True})
 try:
  if target:
   save(PAUSE/'research/paused-for-stable-cup-recovery.json',{'updated_utc':utc(),'saved_environment_steps':target,'resume_when_gpu_slot_available':True,'checkpoint_files_retained':True})
   code="""import os,signal
from pathlib import Path
found=[]
for p in Path('/proc').glob('[0-9]*/cmdline'):
 try:a=p.read_bytes().split(b'\\0')
 except OSError:continue
 if len(a)>2 and a[1]==b'-u' and a[2]==b'/episode/scripts/train_coherent_actor_transfer.py':found.append(int(p.parent.name))
assert len(found)==1,found
os.kill(found[0],signal.SIGTERM)
print('Episode30 yielded its slot after a complete preserved checkpoint',found[0])
"""
   print(_docker(['exec','v2d-grasp-episode30-newton','python','-c',code]).stdout,flush=True)
  state('starting',yielded_episode=30 if target else None,preserved_episode30_environment_steps=target)
  stop=threading.Event()
  threads=[threading.Thread(target=monitor_phase,args=(stop,),daemon=True),threading.Thread(target=previews,args=(stop,),daemon=True)]
  for t in threads:t.start()
  result=None
  try:
   for attempt in range(2):
    try:result=run(10,'stable_contact_20261020');break
    except Exception:print('Stable cup recovery attempt failed',attempt+1,traceback.format_exc(),flush=True)
  finally:stop.set()
  assert result is not None,'Both bounded stable-contact continuation attempts failed'
  save(OUT/'episode_000010-stable-fresh-result.json',result);state('completed',measured_improvement=result.get('measured_improvement',False))
 except Exception:
  state('failed',error=traceback.format_exc());raise
 finally:restore_priority()
