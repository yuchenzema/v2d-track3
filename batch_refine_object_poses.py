"""Audit and retain bounded source-image/collision pose refinement of untrained episodes."""
from pathlib import Path
import fcntl,json,shutil,subprocess,sys,time
from batch_worker import context,BASE,PY
from batch_native import containers,docker

def penalty(a):return sum(sum(max(0.,x['penetration_mm']-3)**2 for x in f['penetrating_cross_body_contacts']) for f in a['frames'])
def atomic_copy(source,destination):
 temporary=destination.with_name(destination.name+'.object-pose-restore.tmp');shutil.copy2(source,temporary);temporary.replace(destination)
for ep in map(int,sys.argv[1:]):
 r,c=context(ep)
 with (r/'research/collision-reference.lock').open('a') as lock:
  fcntl.flock(lock,fcntl.LOCK_EX);assert not (r/'coherent-input-checksums.json').exists()
  selection_path=r/'research/collision-reference-selection.json';selection=json.loads(selection_path.read_text())
  if selection.get('object_6d_refinement_complete'):continue
  backup=r/f'research/object-6d-refinement-{time.time_ns()}';backup.mkdir();path=next((r/'human_motion_data/track3_collision_variant/processed').rglob('motion.parquet'));audit=r/'evaluation/track3_collision_variant-collision-audit.json';shutil.copy2(path,backup/'motion.parquet');shutil.copy2(audit,backup/'audit.json');shutil.copy2(selection_path,backup/'selection.json');before=json.loads(audit.read_text())
  def assess():subprocess.run([PY,str(BASE/'assess_physical_reference.py'),'--run',str(r)],check=True)
  assess();fidelity=r/'evaluation/physical-reference-source-fidelity.json';shutil.copy2(fidelity,backup/'source-fidelity.json');old_source=json.loads(fidelity.read_text())['aggregate']
  if ep in [14,15,16,28]:c['observed_stationary_objects']=[c['objects'].index('plastic_dish_rack')];(r/'episode-config.json').write_text(json.dumps(c,indent=2))
  newton,retarget=containers(r,ep)
  for name in ['refine_native_object_poses','native_reference_audit','optimize_native_grasps','apply_collision_transitions']:shutil.copy2(BASE/f'{name}.py',r/f'scripts/{name}.py')
  def native(script,label,**env):docker(['exec',newton,'env',*[f'{k}={v}' for k,v in env.items()],'python','-u',f'/episode/scripts/{script}.py'],r/f'logs/{label}.log')
  try:
   native('refine_native_object_poses','native-object-6d-refinement')
   native('native_reference_audit','object-6d-collision-audit',V2D_AUDIT_REFERENCE='track3_collision_variant');a=json.loads(audit.read_text())
   if any(any(any('sharpa_wave' in n for n in hit['bodies']) and hit['penetration_mm']>5 for hit in frame['penetrating_cross_body_contacts']) for frame in a['frames']):
    native('optimize_native_grasps','object-6d-hand-clearance-fit',V2D_GRASP_REFERENCE='track3_collision_variant',V2D_OPTIMIZE_CONTACT_FRAMES='1');docker(['exec',retarget,'/isaac-sim/python.sh','-u','/episode/scripts/apply_collision_transitions.py'],r/'logs/object-6d-hand-clearance-apply.log');native('native_reference_audit','object-6d-collision-audit',V2D_AUDIT_REFERENCE='track3_collision_variant')
   after=json.loads(audit.read_text());assess();new_source=json.loads(fidelity.read_text())['aggregate'];moving_image_gain=any(new_source[n]['mean_iou']>old_source[n]['mean_iou']+.005 for b,n in enumerate(c['objects']) if b not in c.get('observed_stationary_objects',[]));accepted=(penalty(after)<penalty(before) or (penalty(after)<=penalty(before)+1e-6 and moving_image_gain)) and all(new_source[n]['mean_iou']>=old_source[n]['mean_iou']-.08 for n in c['objects'])
   if not accepted:
    shutil.copy2(path,backup/'rejected-motion.parquet');shutil.copy2(audit,backup/'rejected-audit.json');shutil.copy2(fidelity,backup/'rejected-source-fidelity.json');atomic_copy(backup/'motion.parquet',path);atomic_copy(backup/'audit.json',audit);atomic_copy(backup/'source-fidelity.json',fidelity);after=before
   selection=json.loads((backup/'selection.json').read_text());selection.update(object_6d_refinement_complete=True,object_6d_refinement_accepted=accepted,object_6d_refinement_backup=str(backup.relative_to(r)),contact_refinement_complete=True,after_frames_over_10mm=after['source_frames_over_10mm'],final_maximum_penetration_mm=after['maximum_cross_body_penetration_mm'],physical_reference_source_fidelity=json.loads(fidelity.read_text())['aggregate']);selection_path.write_text(json.dumps(selection,indent=2));print(ep,'6D refinement',accepted,'penetration penalty',penalty(before),'->',penalty(after),'deep frames',after['source_frames_over_10mm'],flush=True)
  except BaseException:
   atomic_copy(backup/'motion.parquet',path);atomic_copy(backup/'audit.json',audit);atomic_copy(backup/'selection.json',selection_path);atomic_copy(backup/'source-fidelity.json',fidelity);raise
