"""Finish reference reviews before allowing the queued policy learner to freeze its inputs."""
import fcntl,json,shutil,subprocess,sys,time
from pathlib import Path
from batch_worker import context,BASE,PY
episodes=list(map(int,sys.argv[1:]));attempts={ep:0 for ep in episodes}
def active_worker(ep):
 for path in Path('/proc').glob('[0-9]*/cmdline'):
  try:
   argv=path.read_bytes().split(b'\0')
   if argv and Path(argv[0].decode()).name.startswith('python') and b'batch_release_refinement.py' in argv and str(ep).encode() in argv:return True
  except (OSError,UnicodeError):pass
 return False
while episodes:
 for ep in episodes.copy():
  r,c=context(ep);marker=r/'research/continuous-release-selection-v2.json'
  if not marker.exists():
   if not active_worker(ep) and attempts[ep]<3:
    attempts[ep]+=1;log=r/f'logs/release-reference-review-retry-{attempts[ep]}.log'
    with log.open('w') as f:subprocess.Popen([sys.executable,'-u',str(BASE/'batch_release_refinement.py'),str(ep)],cwd=BASE,stdout=f,stderr=subprocess.STDOUT,start_new_session=True)
   continue
  with (r/'research/collision-reference.lock').open('a') as lock:
   fcntl.flock(lock,fcntl.LOCK_EX);assert not (r/'coherent-input-checksums.json').exists();decision=json.loads(marker.read_text());reference=c.get('training_reference','track3_coherent_variant');invariants_path=r/'evaluation/final-release-reference-invariants.json'
   subprocess.run([PY,str(BASE/'verify_release_variant.py'),'--run',str(r),'--baseline',str(r/decision['backup']/'motion.parquet'),'--candidate',decision['candidate_reference'],'--output',str(invariants_path)],check=True);invariants=json.loads(invariants_path.read_text());valid=invariants['objects_bitwise_identical'] and invariants['held_wrist_and_fingers_bitwise_identical'] and invariants['all_releases_continuous']
   if decision['accepted'] and not valid:
    assert not (r/'coherent-input-checksums.json').exists()
    path=next((r/f'human_motion_data/{reference}').rglob('motion.parquet'));audit=r/f'evaluation/{reference}-collision-audit.json'
    for source,dest in [(r/decision['backup']/'motion.parquet',path),(r/decision['backup']/'audit.json',audit)]:
     tmp=dest.with_name(dest.name+'.release-rollback.tmp');shutil.copy2(source,tmp);tmp.replace(dest)
    decision['accepted']=False;decision['rejected_after_final_invariants']=True
   decision.update(pose_invariants_and_continuity_guard_passed=valid,pose_invariants_and_continuity=invariants,review_finalized=True);marker.write_text(json.dumps(decision,indent=2));selection=r/'research/collision-reference-selection.json';value=json.loads(selection.read_text());value.update(continuous_release_review_complete=True,continuous_release_refinement_accepted=decision['accepted'],continuous_release_review=str(marker.relative_to(r)));selection.write_text(json.dumps(value,indent=2));c=json.loads((r/'episode-config.json').read_text());c['reference_review_pending']=False;tmp=r/'episode-config.json.tmp';tmp.write_text(json.dumps(c,indent=2));tmp.replace(r/'episode-config.json');episodes.remove(ep);print(ep,'finalized continuous release',decision['accepted'],'remaining',episodes,flush=True)
 time.sleep(10 if episodes else 0)
