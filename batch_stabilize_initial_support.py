"""Back up, fit and audit source-observed passive rests before freezing training inputs."""
import fcntl,json,shutil,subprocess,sys,time
from batch_worker import context,BASE,PY
from batch_native import containers,docker

for ep in map(int,sys.argv[1:]):
 r,c=context(ep)
 with (r/'research/collision-reference.lock').open('a') as lock:
  fcntl.flock(lock,fcntl.LOCK_EX)
  assert not (r/'coherent-input-checksums.json').exists()
  reference=c.get('training_reference','track3_coherent_variant')
  marker=r/f'research/{reference}-initial-support-selection.json'
  if marker.exists() and json.loads(marker.read_text()).get('accepted'):continue
  path=next((r/f'human_motion_data/{reference}/processed').rglob('motion.parquet'))
  audit=r/f'evaluation/{reference}-collision-audit.json'
  fidelity=r/'evaluation/physical-reference-source-fidelity.json'
  selection=r/'research/collision-reference-selection.json'
  if not fidelity.exists():subprocess.run([PY,str(BASE/'assess_physical_reference.py'),'--run',str(r),'--reference',reference],check=True)
  backup=r/f'research/initial-support-{time.time_ns()}';backup.mkdir()
  saved={path:'motion.parquet',audit:'audit.json',fidelity:'fidelity.json',selection:'selection.json'}
  for original,name in saved.items():
   if original.exists():shutil.copy2(original,backup/name)
  def restore():
   for original,name in saved.items():
    if (backup/name).exists():
     temporary=original.with_name(original.name+'.initial-support-restore.tmp');shutil.copy2(backup/name,temporary);temporary.replace(original)
  before=json.loads(fidelity.read_text())['aggregate']
  newton,retarget=containers(r,ep)
  for name in ['native_reference_audit','optimize_native_grasps','apply_collision_transitions']:
   shutil.copy2(BASE/f'{name}.py',r/f'scripts/{name}.py')
  def native(script,label,**env):
   docker(['exec',newton,'env',*[f'{k}={v}' for k,v in env.items()],'python','-u',f'/episode/scripts/{script}.py'],r/f'logs/{label}.log')
  accepted=False
  try:
   subprocess.run([PY,str(BASE/'stabilize_initial_support.py'),'--run',str(r)],check=True)
   native('native_reference_audit','initial-support-collision-audit',V2D_AUDIT_REFERENCE=reference)
   a=json.loads(audit.read_text())
   if any(any(any('sharpa_wave' in n for n in hit['bodies']) and hit['penetration_mm']>5 for hit in frame['penetrating_cross_body_contacts']) for frame in a['frames']):
    native('optimize_native_grasps','initial-support-hand-clearance',V2D_GRASP_REFERENCE=reference,V2D_OPTIMIZE_CONTACT_FRAMES='1')
    docker(['exec',retarget,'env',f'V2D_TRANSITION_REFERENCE={reference}','/isaac-sim/python.sh','-u','/episode/scripts/apply_collision_transitions.py'],r/'logs/initial-support-hand-apply.log')
    native('native_reference_audit','initial-support-collision-audit',V2D_AUDIT_REFERENCE=reference)
   subprocess.run([PY,str(BASE/'assess_physical_reference.py'),'--run',str(r),'--reference',reference,'--overlay'],check=True)
   after=json.loads(fidelity.read_text())['aggregate'];a=json.loads(audit.read_text())
   accepted=all(after[n]['mean_iou']>=before[n]['mean_iou']-.08 for n in c['objects']) and a['maximum_cross_body_penetration_mm']<15
   decision={'accepted':accepted,'reference':reference,'backup':str(backup.relative_to(r)),'source_fidelity_before':before,'source_fidelity_after':after,'maximum_cross_body_penetration_mm':a['maximum_cross_body_penetration_mm'],'passive_stability_simulation_still_required':True}
   if accepted:
    value=json.loads(selection.read_text());value.update(initial_support_refinement=decision,after_frames_over_10mm=a['source_frames_over_10mm'],final_maximum_penetration_mm=a['maximum_cross_body_penetration_mm']);selection.write_text(json.dumps(value,indent=2))
   else:
    shutil.copy2(path,backup/'rejected-motion.parquet')
    restore()
   marker.write_text(json.dumps(decision,indent=2));print(ep,'initial support accepted',accepted,flush=True)
  except BaseException:
   restore()
   raise
