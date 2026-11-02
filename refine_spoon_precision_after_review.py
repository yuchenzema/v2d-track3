"""Continue a saved spoon learner with tighter placement reward if its review needs it."""
import fcntl,hashlib,json,shutil,time
from pathlib import Path
from improvement_common import BASE,BATCH,baseline,better,save,utc
from batch_native import _docker
R=BASE/'runs/episode_000016_newton_stationary_hold_20261020';OUT=BATCH/'improvements_20261020_contacts';PHASE=OUT/'spoon-precision-phase.json'
with (OUT/'spoon-precision.lock').open('a') as lock:
 fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
 deadline=time.monotonic()+3600;row=None
 while row is None:
  rows=json.loads((R/'checkpoint-reviews.json').read_text())
  row=next((a for a in rows if a['environment_steps']>=374_900_000),None)
  assert time.monotonic()<deadline,'Next spoon checkpoint review did not arrive within an hour'
  if row is None:time.sleep(10)
 if better(row,baseline(16)):
  save(PHASE,{'phase':'skipped_existing_policy_passed_improvement_checks','updated_utc':utc(),'checkpoint_environment_steps':row['environment_steps']})
  print('Spoon checkpoint already improves; preserve its current objective.',flush=True);raise SystemExit(0)
 step=row['environment_steps'];folder=R/'flash_chord_coherent_actor_transfer';archive=R/f'research/precision-stage-checkpoint-{step}';assert not archive.exists();archive.mkdir()
 for kind in ['policy','state']:
  path=folder/f'{kind}_{step}.safetensors';assert path.exists();target=archive/path.name;shutil.copy2(path,target)
  assert hashlib.sha256(path.read_bytes()).hexdigest()==hashlib.sha256(target.read_bytes()).hexdigest()
 for name in ['coherent-training-provenance.json','coherent-training-config.json','episode-config.json']:shutil.copy2(R/name,archive/name)
 adjustment=R/'research/contact-finetune-adjustments.json';shutil.copy2(adjustment,archive/'contact-finetune-adjustments.json')
 metrics=json.loads((folder/'metrics.jsonl').read_text().splitlines()[-1])
 frozen=json.loads((R/'coherent-input-checksums.json').read_text())
 for name,h in frozen.items():assert hashlib.sha256((R/name).read_bytes()).hexdigest()==h,name
 value=json.loads(adjustment.read_text());value.update(updated_utc=utc(),training_translation_precision={'fraction':.25,'scale':.02},
  precision_schedule_checkpoint=step,reason='Review still fails spoon seating/release. Increase centimetre-scale placement reward while retaining the native orientation term and full learner state; evaluation is unchanged.')
 # The current process has already loaded its module. Prepare the next entry
 # point before yielding its slot, so the supervising worker can resume safely.
 shutil.copy2(BASE/'contact_finetune_training.py',R/'scripts/train_coherent_actor_transfer.py')
 shutil.copy2(BASE/'translation_precision_reward.py',R/'scripts/translation_precision_reward.py');save(adjustment,value)
 save(PHASE,{'phase':'resuming_saved_learner_with_position_precision','updated_utc':utc(),'saved_environment_steps':step,
  'last_observed_unsaved_environment_steps':metrics['environment_steps'],'checkpoint_review':row,'full_learner_state_preserved':True,
  'training_only_precision':value['training_translation_precision'],'canonical_evaluation_changed':False,'frozen_model_inputs_unchanged':True})
 code="""import os,signal
from pathlib import Path
found=[]
for p in Path('/proc').glob('[0-9]*/cmdline'):
 try:a=p.read_bytes().split(b'\\0')
 except OSError:continue
 if len(a)>2 and a[1]==b'-u' and a[2]==b'/episode/scripts/train_coherent_actor_transfer.py':found.append(int(p.parent.name))
assert len(found)==1,found
os.kill(found[0],signal.SIGTERM)
print('Spoon precision continuation requested after its preserved checkpoint',found[0])
"""
 print(_docker(['exec','v2d-stationary-episode16-newton','python','-c',code]).stdout,flush=True)
