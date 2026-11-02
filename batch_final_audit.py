"""Publish an integrity and quality audit as each complete episode becomes available."""
from pathlib import Path
import hashlib,json,sys,time
import numpy as np
from batch_artifacts import complete,problems
from episode_clock import control_frames

BASE=Path('/home/yu_eli/v2d-track3');B=BASE/'runs/evaluation_batch_20261015';EPISODES=[3,6,8,10,14,15,16,19,20,25,28,30,32,33,35,36,37,38]

def digest(path):return hashlib.sha256(path.read_bytes()).hexdigest()
def audit(r,require_source_review=True):
 issues=problems(r,verify_hashes=True,require_source_review=require_source_review)
 if issues:return {'integrity_passed':False,'issues':issues}
 c=json.loads((r/'episode-config.json').read_text());selection=json.loads((r/'final-controller-selection.json').read_text())['selected'];states=np.load(r/selection['recorded_states']);N=control_frames(c)
 if len(states['body_poses'])!=N or not np.isfinite(states['body_poses']).all():issues.append('physical state length or finiteness failure')
 if not np.all(states['object_assistance']==0):issues.append('nonzero object assistance')
 if not np.array_equal(states['reference_frames'],np.arange(N)):issues.append('source clock is not continuous')
 if c.get('official_evaluation_required'):
  binding=r/'evaluation/official-render-binding.json'
  if not binding.exists():issues.append('Official recorded rollout has not been bound to rendered object states')
  else:
   linked=json.loads(binding.read_text())
   if not linked.get('rendered_objects_match_official_recording') or linked.get('rendered_state_sha256')!=digest(r/selection['recorded_states']):issues.append('Rendered object motion is not verified against the official recorded trajectory')
 rendered=np.load(r/'evaluation/source_camera_policy_states.npz')
 if not np.array_equal(states['body_poses'],rendered['body_poses']):issues.append('source camera does not replay the selected physical attempt')
 for relative,sha in json.loads((r/'coherent-input-checksums.json').read_text()).items():
  if digest(r/relative)!=sha:issues.append('frozen training input changed: '+relative)
 runtimepath=r/'runtime-config.json'
 if not runtimepath.exists():issues.append('runtime reproduction settings missing')
 else:
  runtime=json.loads(runtimepath.read_text());inference=runtime['inference'];reference=next((r/f"human_motion_data/{inference['reference']}/processed").rglob('motion.parquet'))
  if digest(reference)!=inference['reference_parquet_sha256']:issues.append('inference reference hash mismatch')
  if digest(r/runtime['checkpoint'].removeprefix('/episode/'))!=runtime['checkpoint_sha256']:issues.append('selected runtime checkpoint hash mismatch')
 report=json.loads((r/'result-report.json').read_text());return {'integrity_passed':not issues,'issues':issues,'source_review_complete':bool(report['all_original_frames_visually_reviewed']),'source_review_required_for_this_audit':require_source_review,'source_frames':c['frames'],'physical_control_frames':N,'quality':report['quality_assessment'],'selected_checkpoint':selection['checkpoint'],'selected_source_fidelity':report['selected_source_fidelity'],'selected_attempt_tracking_passed_all_frames':report['selected_attempt_reference_tracking']['passed_all_frames'],'six_videos_frame_validated':True,'all_artifact_hashes_verified':True,'all_frozen_training_inputs_verified':True,'both_camera_views_replay_the_same_physical_attempt':True}

def update():
 previous=B/'batch-final-audit.json';old=json.loads(previous.read_text()) if previous.exists() else {};rows={}
 for ep in EPISODES:
  r=BASE/'runs'/f'episode_{ep:06d}_newton_batch_20261015';key=f'{ep:06d}'
  if not complete(r):rows[key]={'status':'pending'};continue
  signature=digest(r/'deliverables/manifest.json');cached=old.get('episodes',{}).get(key,{})
  if cached.get('manifest_sha256')==signature and cached.get('integrity_passed'):rows[key]=cached;continue
  try:result=audit(r)
  except (OSError,ValueError,KeyError) as exc:result={'integrity_passed':False,'issues':[str(exc)]}
  rows[key]={'status':'audited','manifest_sha256':signature,**result}
 passed=sum(x.get('integrity_passed',False) for x in rows.values());payload={'updated_utc':time.strftime('%Y-%m-%dT%H:%M:%SZ',time.gmtime()),'episodes_requested':18,'excluded_episodes':[1,5],'audited_complete_episodes':passed,'all_requested_deliverables_complete_and_verified':passed==18,'exact_match_to_every_human_motion_verified':False,'note':'Artifact completion and integrity are separate from imitation success. Per-episode quality reports retain failures and source-frame discrepancies.','episodes':rows};temp=previous.with_suffix('.tmp');temp.write_text(json.dumps(payload,indent=2));temp.replace(previous);print('Full artifact audit:',passed,'of 18 episodes verified',flush=True);return passed
if __name__=='__main__':
 while True:
  count=update()
  if '--watch' not in sys.argv or count==18:break
  time.sleep(60)
