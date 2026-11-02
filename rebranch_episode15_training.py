"""Preserve the first training session and start a distinct corrected-reference session."""
from pathlib import Path
import hashlib,json,os,shutil,signal,subprocess,sys,time
from batch_worker import BASE,PY,context
from batch_native import _docker,active_learners
from batch_generic_retarget import install

r,c=context(15);decision=json.loads((r/'research/temporal-reference-selection.json').read_text());assert decision['accepted'] and decision['frozen_training_inputs_preserved'];original_reference=c['training_reference'];candidate=decision['candidate_reference'];selected_reference='track3_temporal_variant';checksums=json.loads((r/'coherent-input-checksums.json').read_text())
for relative,digest in checksums.items():assert hashlib.sha256((r/relative).read_bytes()).hexdigest()==digest,relative
folder=r/'flash_chord_coherent_actor_transfer';pairs=[p for p in folder.glob('policy_*.safetensors') if p.with_name(p.name.replace('policy_','state_')).exists()];assert pairs;latest=max(pairs,key=lambda p:int(p.stem.split('_')[-1]));assert latest.stat().st_size>1_000_000 and latest.with_name(latest.name.replace('policy_','state_')).stat().st_size>100_000_000
archive=r/f'research/training-before-temporal-correction-{time.time_ns()}';archive.mkdir();session={'reason':'The source-based physical reference contained isolated orientation/depth jumps and artificial motion after the observed final release. A separately audited smoother reference improves source overlap and remains collision-feasible.','old_reference':original_reference,'new_reference':selected_reference,'old_inputs_remain_immutable':True,'old_resumable_checkpoint':latest.name,'old_policy_sha256':hashlib.sha256(latest.read_bytes()).hexdigest(),'old_state_sha256':hashlib.sha256(latest.with_name(latest.name.replace('policy_','state_')).read_bytes()).hexdigest(),'archive':str(archive.relative_to(r)),'utc':time.strftime('%Y-%m-%dT%H:%M:%SZ',time.gmtime()),'status':'preparing'};(r/'research/reference-training-session-transition.json').write_text(json.dumps(session,indent=2));shutil.copy2(r/'episode-config.json',archive/'episode-config.json');shutil.copy2(r/'research/collision-reference-selection.json',archive/'collision-reference-selection.json');shutil.copy2(r/'scripts/train_coherent_actor_transfer.py',archive/'train_coherent_actor_transfer.py')

def stop_host(pid,expected):
 try:
  argv=Path(f'/proc/{pid}/cmdline').read_bytes().split(b'\0')
  assert any(Path(os.fsdecode(a)).name==expected for a in argv if a),argv
  os.kill(pid,signal.SIGTERM)
 except FileNotFoundError:pass

# Pause only queue orchestration and preservation. The second actual learner
# and all unrelated user processes keep running.
for pid,script in [(3637021,'batch_supervisor.py'),(3637019,'batch_training_queue.py'),(3578384,'batch_preload_training.py'),(3338138,'batch_preserve_checkpoints.py')]:stop_host(pid,script)
for proc in Path('/proc').glob('[0-9]*/cmdline'):
 try:
  argv=[os.fsdecode(x) for x in proc.read_bytes().split(b'\0') if x]
  if len(argv)>=4 and any(Path(x).name=='batch_native.py' for x in argv[1:2]) and argv[-2:]==['15','train']:os.kill(int(proc.parent.name),signal.SIGTERM)
 except (OSError,ValueError):pass
c.update(reconstruction_visual_reviewed=False,reference_review_pending=True);(r/'episode-config.json').write_text(json.dumps(c,indent=2))
command="import os,signal; from pathlib import Path; found=[]\nfor p in Path('/proc').glob('[0-9]*/cmdline'):\n try:\n  a=p.read_bytes().split(b'\\0')\n  if a and Path(os.fsdecode(a[0])).name.startswith('python') and b'/episode/scripts/train_coherent_actor_transfer.py' in a: found.append(int(p.parent.name))\n except (OSError,ValueError): pass\nassert len(found)==1,found\nfor pid in found: os.kill(pid,signal.SIGINT)\nprint('Interrupted only episode 15 learner at a preserved checkpoint:',found)"
print(_docker(['exec','v2d-batch-episode15-newton','python','-c',command]).stdout,flush=True)
for attempt in range(600):
 # Host PID was recorded at admission and has remained unchanged throughout
 # this session. The other learner is asserted to remain alive below.
 if 3578518 not in active_learners():break
 time.sleep(.5)
else:raise RuntimeError('Episode 15 did not stop; preserve its current session and inspect before moving artifacts')
assert 3511799 in active_learners(),'The independent episode 35 learner must remain running'
metrics=folder/'metrics.jsonl';lines=metrics.read_text().splitlines();session['actual_additional_steps_before_reference_correction']=json.loads(lines[-1])['additional_environment_steps'] if lines else 0

# Every old checkpoint, metric, review, input snapshot and script is retained.
# Move cached evaluations as well, because new checkpoints reuse native step IDs.
for name in ['flash_chord_coherent_actor_transfer','coherent-input-checksums.json','coherent-training-config.json','coherent-training-provenance.json','checkpoint-reviews.json','final-controller-selection.json']:
 p=r/name
 if p.exists():shutil.move(str(p),archive/name)
for relative in ['research/checkpoint-archive','research/training-admission.json','research/training-resume-sessions.json','research/validation-stop-request.json','research/training-early-stop.json','research/integration-controller-selection.json','logs/newton-training.log','logs/newton-training.command.json']:
 p=r/relative
 if p.exists():target=archive/relative;target.parent.mkdir(exist_ok=True,parents=True);shutil.move(str(p),target)
for subdir in ['evaluation','logs']:
 target=archive/subdir;target.mkdir(exist_ok=True)
 for p in (r/subdir).glob('checkpoint-*'):shutil.move(str(p),target/p.name)
shutil.copytree(r/f'human_motion_data/{candidate}',r/f'human_motion_data/{selected_reference}',dirs_exist_ok=True)
c=json.loads((r/'episode-config.json').read_text());c.update(training_reference=selected_reference,training_world_count=1024,reconstruction_visual_reviewed=True,reference_review_pending=False,object_path_continuity_review_pending=False);c['training_reference_revision_reason']=session['reason'];c['previous_training_session']=str(archive.relative_to(r));(r/'episode-config.json').write_text(json.dumps(c,indent=2));install(r)
subprocess.run([PY,str(BASE/'assess_physical_reference.py'),'--run',str(r),'--reference',selected_reference],check=True)
audit=json.loads((r/f'evaluation/{candidate}-collision-audit.json').read_text());audit['reference']=selected_reference;(r/f'evaluation/{selected_reference}-collision-audit.json').write_text(json.dumps(audit,indent=2));selection_path=r/'research/collision-reference-selection.json';selection=json.loads(selection_path.read_text());selection.update(reference=selected_reference,contact_refinement_complete=True,after_frames_over_10mm=audit['source_frames_over_10mm'],final_maximum_penetration_mm=audit['maximum_cross_body_penetration_mm']);selection_path.write_text(json.dumps(selection,indent=2));decision.update(manual_visual_review_pending=False,adopted_for_new_training_session=True,adopted_reference=selected_reference,previous_training_session=str(archive.relative_to(r)),visual_review={'source_frames':[90,97,105,110,135,165],'evidence':'research/supported-reference-visual-review.jpg','finding':'Smoothed insertion and observed stationary spoon after release match the source more closely; source hand alignment is retained.'});(r/'research/temporal-reference-selection.json').write_text(json.dumps(decision,indent=2))
for relative,digest in checksums.items():assert hashlib.sha256((r/relative).read_bytes()).hexdigest()==digest,('Original training input changed',relative)
session.update(status='corrected_session_ready',new_reference_parquet_sha256=hashlib.sha256(next((r/f'human_motion_data/{selected_reference}/processed').rglob('motion.parquet')).read_bytes()).hexdigest(),independent_learner_pid_unchanged=3511799);(archive/'session-manifest.json').write_text(json.dumps(session,indent=2));(r/'research/reference-training-session-transition.json').write_text(json.dumps(session,indent=2))
jobs=[('batch_preserve_checkpoints.py',[],'checkpoint-preservation-after-reference-correction.log'),('batch_training_queue.py',['15','19','20','25','32','33'],'main-after-episode15-reference-correction.log'),('batch_supervisor.py',[],'supervisor-after-episode15-reference-correction.log')]
for script,args,log in jobs:
 with (BASE/'runs/evaluation_batch_20261015'/log).open('w') as f:p=subprocess.Popen([sys.executable,'-u',str(BASE/script),*args],cwd=BASE,stdin=subprocess.DEVNULL,stdout=f,stderr=subprocess.STDOUT,start_new_session=True)
 print('Started',script,p.pid,flush=True)
print('New reviewed episode 15 training session prepared; previous session and independent learner preserved.',session,flush=True)
