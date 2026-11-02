"""Check bounded object-contact correction and release-frame hand clearance before learning."""
from pathlib import Path
import json,fcntl,shutil,sys,time
from batch_worker import context,BASE
from batch_native import containers,docker

def penalty(a):return sum(sum(max(0.,x['penetration_mm']-3)**2 for x in row['penetrating_cross_body_contacts']) for row in a['frames'])
def refine(ep):
 r,c=context(ep)
 with (r/'research/collision-reference.lock').open('a') as lock:
  fcntl.flock(lock,fcntl.LOCK_EX);marker=r/'research/collision-reference-selection.json';selection=json.loads(marker.read_text())
  if selection.get('contact_refinement_complete') or not selection['accepted']:return selection
  assert not (r/'coherent-input-checksums.json').exists(),'Never modify frozen training inputs'
  newton,retarget=containers(r,ep);path=next((r/'human_motion_data/track3_collision_variant').rglob('motion.parquet'));backup=r/f'research/contact-refinement-{time.time_ns()}';backup.mkdir();shutil.copy2(path,backup/'motion.parquet');audit=r/'evaluation/track3_collision_variant-collision-audit.json';before=json.loads(audit.read_text());shutil.copy2(audit,backup/'audit.json')
  for name in ['native_reference_audit','optimize_native_grasps','apply_collision_transitions','project_native_object_contacts']:shutil.copy2(BASE/f'{name}.py',r/f'scripts/{name}.py')
  def native(script,label,**env):docker(['exec',newton,'env',*[f'{k}={v}' for k,v in env.items()],'python','-u',f'/episode/scripts/{script}.py'],r/f'logs/{label}.log')
  if (r/'research/native-object-contact-projection.json').exists():shutil.copy2(r/'research/native-object-contact-projection.json',backup/'object-projection.json')
  native('project_native_object_contacts','final-object-contact-projection')
  native('native_reference_audit','final-contact-audit',V2D_AUDIT_REFERENCE='track3_collision_variant');a=json.loads(audit.read_text())
  if any(any(any('sharpa_wave' in n for n in hit['bodies']) and hit['penetration_mm']>5 for hit in frame['penetrating_cross_body_contacts']) for frame in a['frames']):
   native('optimize_native_grasps','final-hand-clearance-fit',V2D_GRASP_REFERENCE='track3_collision_variant',V2D_OPTIMIZE_CONTACT_FRAMES='1')
   docker(['exec',retarget,'/isaac-sim/python.sh','-u','/episode/scripts/apply_collision_transitions.py'],r/'logs/final-hand-clearance-apply.log')
   native('native_reference_audit','final-contact-audit',V2D_AUDIT_REFERENCE='track3_collision_variant')
  after=json.loads(audit.read_text());accepted=penalty(after)<=penalty(before)+1e-6
  if not accepted:
   shutil.copy2(path,backup/'rejected-motion.parquet');shutil.copy2(audit,backup/'rejected-audit.json');shutil.copy2(backup/'motion.parquet',path);shutil.copy2(backup/'audit.json',audit);after=before
  selection.update(contact_refinement_complete=True,object_contact_refinement_accepted=accepted,contact_penalty_before_refinement=penalty(before),contact_penalty_after_refinement=penalty(after),after_frames_over_10mm=after['source_frames_over_10mm'],final_maximum_penetration_mm=after['maximum_cross_body_penetration_mm'],contact_refinement_backup=str(backup.relative_to(r)));marker.write_text(json.dumps(selection,indent=2));print(ep,'contact refinement',selection,flush=True);return selection
if __name__=='__main__':
 for ep in map(int,sys.argv[1:]):refine(ep)
