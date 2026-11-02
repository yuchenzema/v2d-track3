"""Temporarily devote learner slots to10/16 after preserving33's next checkpoint."""
import fcntl,hashlib,json,time
from pathlib import Path
from batch_native import _docker
from improvement_common import BASE,BATCH,save,utc
P=BATCH/'training-priority.json';F=BATCH/'improvements_20261020_contacts/compute-focus-status.json'
R=BASE/'runs/episode_000033_newton_backward_grasp_20261020'
def focused_root(ep,variant):
 root=BASE/f'runs/episode_{ep:06d}_newton_{variant}_20261020'
 while (root/'research/superseded-training.json').exists():
  target=json.loads((root/'research/superseded-training.json').read_text()).get('superseded_by_run')
  if not target:break
  root=Path(target)
 return root
with (F.with_suffix('.lock')).open('a') as lock:
 fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
 deadline=time.monotonic()+600;checkpoint=None
 while checkpoint is None:
  for path in sorted((R/'grasp_pretraining/stage_3').glob('state_*.safetensors'),key=lambda x:int(x.stem.split('_')[-1]),reverse=True):
   step=int(path.stem.split('_')[-1]);actor=path.with_name(path.name.replace('state_','policy_',1))
   if step<19_000_000 or not actor.exists() or min(time.time()-q.stat().st_mtime for q in [path,actor])<5:continue
   metas=[]
   for file in [path,actor]:
    with file.open('rb') as f:n=int.from_bytes(f.read(8),'little');h=json.loads(f.read(n))
    assert file.stat().st_size==8+n+max(x['data_offsets'][1] for k,x in h.items() if k!='__metadata__');metas.append(h['__metadata__'])
   assert metas[0]['user.checkpoint_set_id']==metas[1]['user.checkpoint_set_id'];checkpoint=path;break
  if checkpoint is not None:break
  assert time.monotonic()<deadline,'No safe saved checkpoint became available'
  save(F,{'phase':'waiting_for_episode33_saved_checkpoint','updated_utc':utc(),'focus_episodes':[10,16]});time.sleep(10)
 priority=json.loads(P.read_text());saved_names=priority.get('allowed_run_names',[]);previous_max=priority.get('maximum_concurrent_learners',3)
 priority.update(maximum_concurrent_learners=2,allowed_run_names=[n for n in saved_names if n.startswith(('episode_000010_','episode_000016_'))],
  deferred_for_episode10_16_compute=[n for n in saved_names if not n.startswith(('episode_000010_','episode_000016_'))],
  reason='Prioritize the two episodes most recently reported as unimproved; other learner checkpoints retained.',updated_utc=utc())
 save(P,priority)
 try:
  record={'updated_utc':utc(),'saved_checkpoint':str(checkpoint),'saved_checkpoint_sha256':hashlib.sha256(checkpoint.read_bytes()).hexdigest(),
   'saved_environment_steps':int(checkpoint.stem.split('_')[-1]),'checkpoint_retained':True,'resume_when_focus_released':True}
  save(R/'research/paused-for-episodes10-16.json',record)
  code="""import os,signal
from pathlib import Path
found=[]
for p in Path('/proc').glob('[0-9]*/cmdline'):
 try:a=p.read_bytes().split(b'\\0')
 except OSError:continue
 if len(a)>2 and a[1]==b'-u' and a[2]==b'/episode/scripts/train_coherent_actor_transfer.py':found.append(int(p.parent.name))
assert len(found)==1,found
os.kill(found[0],signal.SIGTERM)
print('Episode33 paused after its saved full learner checkpoint',found[0])
"""
  print(_docker(['exec','v2d-backward-episode33-newton','python','-c',code]).stdout,flush=True)
  deadline=time.monotonic()+8*3600
  while time.monotonic()<deadline:
   roots=[focused_root(10,'stable_contact'),focused_root(16,'stationary_hold')]
   if all((r/'flash_chord_coherent_actor_transfer/TRAINING_COMPLETED').exists() for r in roots):break
   save(F,{'phase':'focused_training','updated_utc':utc(),'focus_episodes':[10,16],'maximum_concurrent_learners':2,
    'paused_episode33_checkpoint':str(checkpoint),'automatic_priority_restore_after_maximum_hours':8});time.sleep(15)
 finally:
  current=json.loads(P.read_text());current['allowed_run_names']=list(dict.fromkeys(current.get('allowed_run_names',[])+current.pop('deferred_for_episode10_16_compute',[])))
  current['maximum_concurrent_learners']=previous_max;current['updated_utc']=utc();save(P,current)
  save(F,{'phase':'priority_restored','updated_utc':utc(),'focus_episodes':[10,16],'episode33_checkpoint_retained':True})
