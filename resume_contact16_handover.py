"""Resume16 at its reviewed checkpoint with a handover-focused training schedule."""
from pathlib import Path
import hashlib,json,shutil,time
from improvement_common import BASE,BATCH,save,utc
from batch_native import _docker
r=BASE/'runs/episode_000016_newton_grasp_curriculum_20261020';f=r/'flash_chord_coherent_actor_transfer';step=355_000_320
manifest=json.loads((r/'coherent-input-checksums.json').read_text())
for p,h in manifest.items():assert hashlib.sha256((r/p).read_bytes()).hexdigest()==h,('Frozen scene input changed',p)
cp=[f/f'{kind}_{step}.safetensors' for kind in ['policy','state']];meta=[]
for p in cp:
 with p.open('rb') as handle:length=int.from_bytes(handle.read(8),'little');header=json.loads(handle.read(length))
 m={k.removeprefix('user.'):v for k,v in header['__metadata__'].items() if k.startswith('user.')};assert int(m['environment_steps'])==step;meta.append(m)
assert meta[0]['checkpoint_set_id']==meta[1]['checkpoint_set_id']
archive=r/f'research/handover-schedule-checkpoint-{step}';archive.mkdir(exist_ok=True);hashes={}
for p in cp:
 shutil.copy2(p,archive/p.name);hashes[p.name]=hashlib.sha256(p.read_bytes()).hexdigest();assert hashlib.sha256((archive/p.name).read_bytes()).hexdigest()==hashes[p.name]
for name in ['train_coherent_actor_transfer.py','training_finger_reset.py']:shutil.copy2(r/'scripts'/name,archive/name)
adjustment={'updated_utc':utc(),'source_interval':[54,84],'residual_mean_learning_rate_multiplier':.1,
 'reason':'The5M independent review improves pickup/rotation but fails at the right-hand regrasp. Practice the handover before insertion, retaining the original baseline residual-head learning-rate multiplier.',
 'evidence':'research/human-event-checkpoint-reviews.json','scene_geometry_and_reference_unchanged':True,
 'canonical_evaluation_starts_at_source_frame_zero':True,'object_assistance':0.,'policy_improvement_verified':False}
save(r/'research/contact-finetune-adjustments.json',adjustment)
for src,dst in [('contact_finetune_training.py','train_coherent_actor_transfer.py'),('training_finger_reset.py','training_finger_reset.py')]:
 target=r/'scripts'/dst;tmp=target.with_suffix('.resume.tmp');shutil.copy2(BASE/src,tmp);tmp.replace(target)
p=BATCH/'training-priority.json';priority=json.loads(p.read_text());deferred=[n for n in priority.get('allowed_run_names',[])+priority.get('deferred_for_contact_admission_run_names',[]) if any(n.startswith(f'episode_{ep:06d}_') for ep in [19,30,33])]
priority['allowed_run_names']=[n for n in priority.get('allowed_run_names',[]) if n not in deferred];priority['deferred_for_contact_admission_run_names']=[];priority['temporarily_deferred_resume_run_names']=list(dict.fromkeys(deferred));priority['updated_utc']=utc();save(p,priority)
latest=json.loads((f/'metrics.jsonl').read_text().splitlines()[-1]);history=r/'research/training-resume-sessions.json';rows=json.loads(history.read_text()) if history.exists() else []
rows.append({'utc':utc(),'checkpoint_environment_steps':step,'checkpoint_sha256':hashes,'checkpoint_archive':str(archive.relative_to(r)),
 'latest_unsaved_environment_steps':latest['environment_steps'],'unsaved_environment_steps_to_repeat':latest['environment_steps']-step,
 'reason':adjustment['reason'],'optimizer_critic_temperature_and_normalizer_preserved':True,'replay_restarted':True,
 'old_focus_source_interval':[84,134],'new_focus_source_interval':[54,84],
 'old_residual_mean_learning_rate_multiplier':1.,'new_residual_mean_learning_rate_multiplier':.1,'frozen_scene_inputs_unchanged':True})
save(history,rows)
code="""import os,signal
from pathlib import Path
found=[]
for p in Path('/proc').glob('[0-9]*/cmdline'):
 try:a=p.read_bytes().split(b'\\0')
 except OSError:continue
 if len(a)>2 and a[1]==b'-u' and a[2]==b'/episode/scripts/train_coherent_actor_transfer.py':found.append(int(p.parent.name))
assert len(found)==1,found
os.kill(found[0],signal.SIGTERM)
print('Stopped only16 for the documented checkpoint resume',found[0])
"""
print(_docker(['exec','v2d-grasp-episode16-newton','python','-c',code]).stdout,flush=True)
last_time=latest['utc']
while True:
 lines=(f/'metrics.jsonl').read_text().splitlines();current=json.loads(lines[-1]) if lines else {}
 if current.get('utc','')>last_time and current.get('curriculum/contact_focus_source_start')==54 and current.get('curriculum/contact_focus_source_end')==84:break
 time.sleep(10)
priority=json.loads(p.read_text());priority['allowed_run_names']=list(dict.fromkeys(priority.get('allowed_run_names',[])+priority.get('temporarily_deferred_resume_run_names',[])));priority['temporarily_deferred_resume_run_names']=[];priority['updated_utc']=utc();save(p,priority)
save(r/'research/handover-schedule-resume-status.json',{'updated_utc':utc(),'status':'running','checkpoint_environment_steps':step,'focus_source_interval':[54,84],
 'residual_mean_learning_rate_multiplier':.1,'full_state_restored':True,'scene_inputs_unchanged':True,'evaluation_object_assistance':0.})
print('16 resumed with full learner state, lower residual-head learning rate and handover-focused starts.',flush=True)
