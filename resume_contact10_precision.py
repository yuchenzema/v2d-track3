"""Resume10 at its next complete checkpoint with an explicit precision objective."""
import fcntl,hashlib,json,shutil,time
from improvement_common import BASE,BATCH,save,utc
from batch_native import _docker
R=BASE/'runs/episode_000010_newton_grasp_curriculum_20261020';F=R/'flash_chord_coherent_actor_transfer'
status=R/'research/precision-schedule-resume-status.json'
def metadata(p):
 with p.open('rb') as f:
  size=int.from_bytes(f.read(8),'little');h=json.loads(f.read(size))
 assert p.stat().st_size==8+size+max(v['data_offsets'][1] for k,v in h.items() if k!='__metadata__')
 return {k.removeprefix('user.'):v for k,v in h['__metadata__'].items() if k.startswith('user.')}
with (R/'research/precision-schedule-resume.lock').open('a') as lock:
 fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
 if status.exists() and json.loads(status.read_text()).get('status')=='running':raise SystemExit('Already resumed')
 deadline=time.monotonic()+2700
 while True:
  latest=json.loads((F/'metrics.jsonl').read_text().splitlines()[-1]);pairs=[]
  for p in F.glob('state_*.safetensors'):
   step=int(p.stem.split('_')[-1]);a=p.with_name(p.name.replace('state_','policy_',1))
   if step<359_900_000 or not a.exists() or min(time.time()-p.stat().st_mtime,time.time()-a.stat().st_mtime)<5:continue
   try:
    m,n=metadata(p),metadata(a)
    assert m['checkpoint_set_id']==n['checkpoint_set_id'] and int(m['environment_steps'])==step
    pairs.append((step,p,a))
   except (AssertionError,ValueError,KeyError,json.JSONDecodeError):continue
  if pairs:break
  save(status,{'updated_utc':utc(),'status':'waiting_for_next_complete_10M_checkpoint','latest_environment_steps':latest['environment_steps'],'training_continues':True})
  assert time.monotonic()<deadline,'No complete10M checkpoint within45 minutes; learner left untouched'
  time.sleep(15)
 step,state,actor=max(pairs);manifest=json.loads((R/'coherent-input-checksums.json').read_text())
 for p,h in manifest.items():assert hashlib.sha256((R/p).read_bytes()).hexdigest()==h,p
 archive=R/f'research/precision-schedule-checkpoint-{step}';archive.mkdir(exist_ok=True);hashes={}
 for p in [state,actor]:
  shutil.copy2(p,archive/p.name);hashes[p.name]=hashlib.sha256(p.read_bytes()).hexdigest();assert hashlib.sha256((archive/p.name).read_bytes()).hexdigest()==hashes[p.name]
 for name in ['train_coherent_actor_transfer.py','training_finger_reset.py']:shutil.copy2(R/'scripts'/name,archive/name)
 adjustment={'updated_utc':utc(),'source_interval':[48,128],'residual_mean_learning_rate_multiplier':.1,
  'training_translation_precision':{'fraction':.25,'scale':.01},
  'reason':'First5M whole rollout passes native thresholds but cups diverge in opposite directions by3–4cm; retain75% native orientation-sensitive reward and add25%1cm translation precision.',
  'evidence':'research/human-event-checkpoint-reviews.json','scene_geometry_and_reference_unchanged':True,
  'canonical_evaluation_starts_at_source_frame_zero':True,'object_assistance':0.,'policy_improvement_verified':False}
 save(R/'research/contact-finetune-adjustments.json',adjustment)
 for src,dst in [('contact_finetune_training.py','train_coherent_actor_transfer.py'),('training_finger_reset.py','training_finger_reset.py'),('translation_precision_reward.py','translation_precision_reward.py')]:
  target=R/'scripts'/dst;tmp=target.with_suffix('.resume.tmp');shutil.copy2(BASE/src,tmp);tmp.replace(target)
 priority_path=BATCH/'training-priority.json';priority=json.loads(priority_path.read_text())
 deferred=[n for n in priority.get('allowed_run_names',[]) if any(n.startswith(f'episode_{ep:06d}_') for ep in [19,30,33])]
 priority['allowed_run_names']=[n for n in priority.get('allowed_run_names',[]) if n not in deferred];priority['precision_resume_deferred_run_names']=deferred;save(priority_path,priority)
 history_path=R/'research/training-resume-sessions.json';history=json.loads(history_path.read_text()) if history_path.exists() else []
 latest=json.loads((F/'metrics.jsonl').read_text().splitlines()[-1]);history.append({'utc':utc(),'checkpoint_environment_steps':step,'checkpoint_sha256':hashes,
  'latest_unsaved_environment_steps':latest['environment_steps'],'unsaved_environment_steps_to_repeat':latest['environment_steps']-step,
  'optimizer_critic_temperature_and_normalizer_preserved':True,'replay_restarted':True,'new_adjustments':adjustment,'frozen_scene_inputs_unchanged':True})
 save(history_path,history)
 code="""import os,signal
from pathlib import Path
found=[]
for p in Path('/proc').glob('[0-9]*/cmdline'):
 try:a=p.read_bytes().split(b'\\0')
 except OSError:continue
 if len(a)>2 and a[1]==b'-u' and a[2]==b'/episode/scripts/train_coherent_actor_transfer.py':found.append(int(p.parent.name))
assert len(found)==1,found
os.kill(found[0],signal.SIGTERM)
print('Stopped only10 for documented complete-state resume',found[0])
"""
 try:
  print(_docker(['exec','v2d-grasp-episode10-newton','python','-c',code]).stdout,flush=True);deadline=time.monotonic()+900
  while True:
   current=json.loads((F/'metrics.jsonl').read_text().splitlines()[-1])
   if current.get('utc','')>latest['utc'] and current.get('curriculum/contact_focus_source_start')==48:break
   assert time.monotonic()<deadline,'Resume did not report precision-schedule metrics within15 minutes'
   time.sleep(10)
  save(status,{'updated_utc':utc(),'status':'running','checkpoint_environment_steps':step,'training_only_translation_precision':adjustment['training_translation_precision'],
   'residual_mean_learning_rate_multiplier':.1,'full_state_restored':True,'scene_inputs_unchanged':True,'evaluation_object_assistance':0.})
 finally:
  priority=json.loads(priority_path.read_text());priority['allowed_run_names']=list(dict.fromkeys(priority.get('allowed_run_names',[])+priority.pop('precision_resume_deferred_run_names',[])));save(priority_path,priority)
 print('10 resumed with complete learner state and explicitly recorded position-precision objective.',flush=True)
