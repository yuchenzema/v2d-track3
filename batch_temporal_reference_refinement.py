"""Keep temporally regularized physical references only after independent audits.

Frozen learners receive a separate candidate only. Their current inputs, checkpoints,
training configuration and reference remain unchanged.
"""
import fcntl,hashlib,json,os,shutil,subprocess,sys,time
from batch_worker import BASE,PY,context
from batch_native import containers,docker

def atomic_copy(source,destination):
 tmp=destination.with_suffix('.temporal.tmp');shutil.copy2(source,tmp);tmp.replace(destination)

for ep in map(int,sys.argv[1:]):
 r,c=context(ep)
 with (r/'research/collision-reference.lock').open('a') as lock:
  fcntl.flock(lock,fcntl.LOCK_EX);frozen=(r/'coherent-input-checksums.json').exists();reference=c['training_reference'];candidate='track3_temporal_physical_candidate_v1';source=r/f'human_motion_data/{reference}';destination=r/f'human_motion_data/{candidate}'
  if os.environ.get('V2D_TEMPORAL_CONTINUE_CANDIDATE')!='1':shutil.copytree(source,destination,dirs_exist_ok=True)
  backup=r/f'research/temporal-reference-{time.time_ns()}';backup.mkdir();path=next((source/'processed').rglob('motion.parquet'));shutil.copy2(path,backup/'motion.parquet');hulls=r/'evaluation/object-collision-hulls.npz';shutil.copy2(hulls,backup/'object-collision-hulls.npz');checksums=json.loads((r/'coherent-input-checksums.json').read_text()) if frozen else {};native,retarget=containers(r,ep)
  def verify_frozen():
   for relative,digest in checksums.items():assert hashlib.sha256((r/relative).read_bytes()).hexdigest()==digest,relative
  verify_frozen()
  for script in ['native_reference_audit','refine_native_object_poses','optimize_native_grasps','apply_collision_transitions']:shutil.copy2(BASE/f'{script}.py',r/f'scripts/{script}.py')
  def cpu(script,args):subprocess.run([PY,str(BASE/script),*args],check=True)
  def native_run(script,label,**env):docker(['exec',native,'env','OMP_NUM_THREADS=4',*[f'{k}={v}' for k,v in env.items()],'python','-u',f'/episode/scripts/{script}.py'],r/f'logs/{label}.log')
  def assess(ref,label):
   cpu('assess_physical_reference.py',['--run',str(r),'--reference',ref,'--output',label,'--no-update-selection']);cpu('assess_retarget_hand_projection.py',['--run',str(r),'--reference',ref,'--output',str(r/f'evaluation/{label}-hands.json')]);return json.loads((r/f'evaluation/{label}.json').read_text()),json.loads((r/f'evaluation/{label}-hands.json').read_text())
  try:
   before,before_hands=assess(reference,'temporal-baseline-source');cpu('regularize_object_reference.py',['--run',str(r),'--reference',candidate]);regularization=json.loads((r/'research/object-reference-temporal-candidate.json').read_text());native_run('native_reference_audit','temporal-candidate-audit',V2D_AUDIT_REFERENCE=candidate);auditpath=r/f'evaluation/{candidate}-collision-audit.json';audit=json.loads(auditpath.read_text());objects=[v['object_index'] for v in regularization['objects']]
   if objects and any(any(hit['penetration_mm']>4 and not any('sharpa_wave' in body for body in hit['bodies']) for hit in frame['penetrating_cross_body_contacts']) for frame in audit['frames']):
    start=min(regularization['changed_frames']);end=max(regularization['changed_frames']);end=min(end,{15:109,16:127,30:239}.get(ep,c['frames']-2));native_run('refine_native_object_poses','temporal-object-contact-projection',V2D_OBJECT_POSE_REFERENCE=candidate,V2D_PRESERVE_STATIC_POSES='1',V2D_OBJECT_POSE_MOBILE_OBJECTS=json.dumps(objects),V2D_OBJECT_POSE_START_FRAME=str(start),V2D_OBJECT_POSE_END_FRAME=str(end),V2D_OBJECT_POSE_TEMPORAL_GUARD='1',V2D_OBJECT_POSE_MAX_TRANSLATION=os.environ.get('V2D_TEMPORAL_TRANSLATION_LIMIT','.06'),V2D_OBJECT_POSE_MAX_ROTATION=os.environ.get('V2D_TEMPORAL_ROTATION_LIMIT','.2'),V2D_OBJECT_POSE_OUTPUT='research/temporal-object-contact-projection.json');native_run('native_reference_audit','temporal-candidate-audit',V2D_AUDIT_REFERENCE=candidate);audit=json.loads(auditpath.read_text())
   if any(any(hit['penetration_mm']>5 and any('sharpa_wave' in body for body in hit['bodies']) for hit in frame['penetrating_cross_body_contacts']) for frame in audit['frames']):
    native_run('optimize_native_grasps','temporal-hand-clearance',V2D_GRASP_REFERENCE=candidate,V2D_OPTIMIZE_CONTACT_FRAMES='1',V2D_SOURCE_HAND_SIGMA='16',V2D_CONTACT_FIT_OUTPUT='research/temporal-hand-clearance.json');docker(['exec',retarget,'env',f'V2D_TRANSITION_REFERENCE={candidate}','V2D_CONTACT_FIT_OUTPUT=research/temporal-hand-clearance.json','V2D_CONTACT_APPLY_OUTPUT=research/temporal-hand-clearance-applied.json','/isaac-sim/python.sh','-u','/episode/scripts/apply_collision_transitions.py'],r/'logs/temporal-hand-clearance-apply.log');native_run('native_reference_audit','temporal-candidate-audit',V2D_AUDIT_REFERENCE=candidate);audit=json.loads(auditpath.read_text())
   after,after_hands=assess(candidate,'temporal-candidate-source');result=subprocess.run([PY,str(BASE/'verify_temporal_reference.py'),'--run',str(r),'--reference',candidate,'--baseline',reference],capture_output=True,text=True,check=True);continuity=json.loads(result.stdout);physics_ok=audit['maximum_cross_body_penetration_mm']<=max(10.,json.loads((r/f'evaluation/{reference}-collision-audit.json').read_text())['maximum_cross_body_penetration_mm']+1) and audit['source_frames_over_10mm']<=2;image_ok=all(after['aggregate'][n]['mean_iou']>=before['aggregate'][n]['mean_iou']-.05 for n in c['objects']);hands_ok=all(new['all_mean_landmark_error_px']<=old['all_mean_landmark_error_px']*1.15+8 and new['all_mean_wrist_error_px']<=old['all_mean_wrist_error_px']*1.2+8 for old,new in zip(before_hands['aggregate'],after_hands['aggregate']));accepted=physics_ok and image_ok and hands_ok and continuity['passed']
   if accepted and not frozen:
    atomic_copy(next((destination/'processed').rglob('motion.parquet')),path);audit['reference']=reference;out=r/f'evaluation/{reference}-collision-audit.json';tmp=out.with_suffix('.temporal.tmp');tmp.write_text(json.dumps(audit,indent=2));tmp.replace(out);after['reference']=reference;out=r/'evaluation/physical-reference-source-fidelity.json';tmp=out.with_suffix('.temporal.tmp');tmp.write_text(json.dumps(after,indent=2));tmp.replace(out);selection=r/'research/collision-reference-selection.json';value=json.loads(selection.read_text());value.update(physical_reference_source_fidelity=after['aggregate'],after_frames_over_10mm=audit['source_frames_over_10mm'],final_maximum_penetration_mm=audit['maximum_cross_body_penetration_mm']);selection.write_text(json.dumps(value,indent=2))
   decision={'method':__doc__,'accepted':accepted,'canonical_reference_updated':accepted and not frozen,'frozen_training_inputs_preserved':frozen,'reference':reference,'candidate_reference':candidate,'source_before':before['aggregate'],'source_after':after['aggregate'],'hand_projection_before':before_hands['aggregate'],'hand_projection_after':after_hands['aggregate'],'collision_guard_passed':physics_ok,'image_guard_passed':image_ok,'hand_image_guard_passed':hands_ok,'continuity_guard':continuity,'maximum_cross_body_penetration_mm':audit['maximum_cross_body_penetration_mm'],'source_frames_over_10mm':audit['source_frames_over_10mm'],'backup':str(backup.relative_to(r)),'manual_visual_review_pending':True};(r/'research/temporal-reference-selection.json').write_text(json.dumps(decision,indent=2));print(ep,'TEMPORAL REFERENCE',decision,flush=True)
  finally:
   if frozen:atomic_copy(backup/'object-collision-hulls.npz',hulls);verify_frozen()
