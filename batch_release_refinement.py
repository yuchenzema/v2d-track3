"""Review continuous source-derived release paths before admitting them to training."""
import fcntl,json,shutil,subprocess,sys,time,os
from batch_worker import context,BASE,PY
from batch_native import containers,docker

def atomic_copy(source,destination):
 temporary=destination.with_name(destination.name+'.release-refinement.tmp');shutil.copy2(source,temporary);temporary.replace(destination)

for ep in map(int,sys.argv[1:]):
 r,c=context(ep)
 with (r/'research/collision-reference.lock').open('a') as lock:
  fcntl.flock(lock,fcntl.LOCK_EX);inference_only=os.environ.get('V2D_RELEASE_INFERENCE_ONLY')=='1';assert not (r/'coherent-input-checksums.json').exists() or inference_only
  version=os.environ.get('V2D_RELEASE_REFINEMENT_VERSION','2');suffix='-inference' if inference_only else '';marker=r/f'research/continuous-release-selection-v{version}{suffix}.json'
  if marker.exists():continue
  reference=c.get('training_reference','track3_coherent_variant');candidate=reference+f'_continuous_release_candidate_v{version}';folder=r/f'human_motion_data/{reference}';destination=r/f'human_motion_data/{candidate}';shutil.copytree(folder,destination,dirs_exist_ok=True);path=next((folder/'processed').rglob('motion.parquet'));candidate_path=next((destination/'processed').rglob('motion.parquet'));backup=r/f'research/continuous-release-{time.time_ns()}';backup.mkdir();shutil.copy2(path,backup/'motion.parquet');audit=r/f'evaluation/{reference}-collision-audit.json';shutil.copy2(audit,backup/'audit.json');before=json.loads(audit.read_text());newton,retarget=containers(r,ep)
  for name in ['continuous_release_reference','native_reference_audit','optimize_native_grasps','apply_collision_transitions']:shutil.copy2(BASE/f'{name}.py',r/f'scripts/{name}.py')
  def project(ref,label):
   output=r/f'evaluation/{label}-hand-projection.json';subprocess.run([PY,str(BASE/'assess_retarget_hand_projection.py'),'--run',str(r),'--reference',ref,'--output',str(output)],check=True);return json.loads(output.read_text())
  def native(script,label,**env):docker(['exec',newton,'env',*[f'{k}={v}' for k,v in env.items()],'python','-u',f'/episode/scripts/{script}.py'],r/f'logs/{label}.log')
  old_hands=project(reference,'before-continuous-release')
  docker(['exec',retarget,'env',f'V2D_CONTINUOUS_REFERENCE={candidate}',*(['V2D_ALLOW_FROZEN_ALTERNATE=1'] if inference_only else []),*(['V2D_RELEASE_PATH_MODEL=pinhole'] if int(version)>=3 else []),'/isaac-sim/python.sh','-u','/episode/scripts/continuous_release_reference.py'],r/'logs/continuous-release-fit.log')
  native('native_reference_audit','continuous-release-audit',V2D_AUDIT_REFERENCE=candidate);candidate_audit=r/f'evaluation/{candidate}-collision-audit.json';a=json.loads(candidate_audit.read_text());free={side:[i for i in range(c['frames']) if not any(g['side']==side and g['start']<=i<=g['end'] for g in c['grasps'])] for side in ['left','right']};fit='research/continuous-release-clearance-fit.json'
  if any(any(hit['penetration_mm']>5 and any(n.startswith(side+'_sharpa_wave_') for n in hit['bodies']) and frame['source_frame'] in free[side] for hit in frame['penetrating_cross_body_contacts'] for side in ['left','right']) for frame in a['frames']):
   native('optimize_native_grasps','continuous-release-clearance-fit',V2D_GRASP_REFERENCE=candidate,V2D_OPTIMIZE_CONTACT_FRAMES='1',V2D_CONTACT_FRAME_SELECTION=json.dumps(free),V2D_CONTACT_FIT_OUTPUT=fit)
   docker(['exec',retarget,'env',f'V2D_TRANSITION_REFERENCE={candidate}',f'V2D_CONTACT_FIT_OUTPUT={fit}','V2D_CONTACT_APPLY_OUTPUT=research/continuous-release-clearance-applied.json','/isaac-sim/python.sh','-u','/episode/scripts/apply_collision_transitions.py'],r/'logs/continuous-release-clearance-apply.log')
   native('native_reference_audit','continuous-release-audit',V2D_AUDIT_REFERENCE=candidate);a=json.loads(candidate_audit.read_text())
  new_hands=project(candidate,'after-continuous-release');image_ok=all(new['free_mean_landmark_error_px'] is None or new['free_mean_landmark_error_px']<=old['free_mean_landmark_error_px']*1.1+8 for old,new in zip(old_hands['aggregate'],new_hands['aggregate']));physics_ok=a['source_frames_over_10mm']<=before['source_frames_over_10mm']+2 and a['maximum_cross_body_penetration_mm']<=max(12.,before['maximum_cross_body_penetration_mm']+1.)
  continuity_path=r/f'evaluation/{candidate}-invariants.json';subprocess.run([PY,str(BASE/'verify_release_variant.py'),'--run',str(r),'--baseline',str(backup/'motion.parquet'),'--candidate',candidate,'--output',str(continuity_path)],check=True);invariants=json.loads(continuity_path.read_text());continuity_ok=invariants['objects_bitwise_identical'] and invariants['held_wrist_and_fingers_bitwise_identical'] and invariants['all_releases_continuous'];accepted=image_ok and physics_ok and continuity_ok
  if accepted and not inference_only:
   atomic_copy(candidate_path,path);a['reference']=reference;temporary=audit.with_suffix('.host.tmp');temporary.write_text(json.dumps(a,indent=2));temporary.replace(audit)
  decision={'accepted':accepted,'inference_only_alternate':inference_only,'frozen_training_inputs_preserved':inference_only,'criterion':'Source-derived continuous free-hand motion; held poses and all objects fixed; accept only without material image alignment or static contact regression. Actual learned-policy evaluation remains required.','candidate_reference':candidate,'backup':str(backup.relative_to(r)),'source_projection_before':old_hands['aggregate'],'source_projection_after':new_hands['aggregate'],'source_projection_guard_passed':image_ok,'static_collision_guard_passed':physics_ok,'pose_invariants_and_continuity_guard_passed':continuity_ok,'pose_invariants_and_continuity':invariants,'maximum_cross_body_penetration_before_mm':before['maximum_cross_body_penetration_mm'],'maximum_cross_body_penetration_after_mm':a['maximum_cross_body_penetration_mm'],'source_frames_over_10mm_after':a['source_frames_over_10mm']};marker.write_text(json.dumps(decision,indent=2));print(ep,'continuous release accepted',accepted,decision,flush=True)
