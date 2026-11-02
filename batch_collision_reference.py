"""Build and audit physically feasible grasp references before starting each learner."""
import json,shutil,subprocess,sys,fcntl
from pathlib import Path
from batch_native import containers,docker
from batch_worker import context,BASE,PY

def _prepare(ep):
 r,c=context(ep);marker=r/'research/collision-reference-selection.json'
 if marker.exists():return json.loads(marker.read_text())
 assert not (r/'coherent-input-checksums.json').exists(),'Never change the reference of an active or completed learner'
 newton,retarget=containers(r,ep)
 for name in ['native_reference_audit','optimize_native_grasps','apply_collision_grasps','apply_collision_transitions','project_native_object_contacts']:shutil.copy2(BASE/f'{name}.py',r/f'scripts/{name}.py')
 def native(script,label,**env):docker(['exec',newton,'env',*[f'{k}={v}' for k,v in env.items()],'python','-u',f'/episode/scripts/{script}.py'],r/f'logs/{label}.log')
 def apply(script,label):docker(['exec',retarget,'/isaac-sim/python.sh','-u',f'/episode/scripts/{script}.py'],r/f'logs/{label}.log')
 native('native_reference_audit','collision-coherent-audit')
 native('optimize_native_grasps','collision-grasp-fit')
 apply('apply_collision_grasps','collision-grasp-apply')
 native('native_reference_audit','collision-stable-audit',V2D_AUDIT_REFERENCE='track3_collision_variant')
 audit=r/'evaluation/track3_collision_variant-collision-audit.json';shutil.copy2(audit,r/'evaluation/collision-before-transitions.json')
 native('optimize_native_grasps','collision-transition-fit',V2D_GRASP_REFERENCE='track3_collision_variant',V2D_OPTIMIZE_CONTACT_FRAMES='1')
 apply('apply_collision_transitions','collision-transition-apply')
 subprocess.run([PY,str(BASE/'project_collision_floor.py'),'--run',str(r)],check=True)
 native('project_native_object_contacts','project-native-object-contacts')
 native('native_reference_audit','collision-final-audit',V2D_AUDIT_REFERENCE='track3_collision_variant')
 final=json.loads(audit.read_text())
 if any(any(any('sharpa_wave' in n for n in hit['bodies']) and hit['penetration_mm']>5 for hit in frame['penetrating_cross_body_contacts']) for frame in final['frames']):
  native('optimize_native_grasps','collision-final-transition-fit',V2D_GRASP_REFERENCE='track3_collision_variant',V2D_OPTIMIZE_CONTACT_FRAMES='1')
  apply('apply_collision_transitions','collision-final-transition-apply')
  native('native_reference_audit','collision-final-audit',V2D_AUDIT_REFERENCE='track3_collision_variant')
 before=json.loads((r/'evaluation/track3_coherent_variant-collision-audit.json').read_text());after=json.loads(audit.read_text())
 def magnitude(a):return sum(sum(max(0.,v['penetration_mm']-3)**2 for v in row['penetrating_cross_body_contacts']) for row in a['frames'])
 improved=magnitude(after)<magnitude(before);reference='track3_collision_variant' if improved else 'track3_coherent_variant'
 out={'reference':reference,'accepted':improved,'criterion':'Reduced sum of squared cross-body penetration above 3 mm over every source frame; original source trajectories retained separately.','penalty_before':magnitude(before),'penalty_after':magnitude(after),'before_frames_over_10mm':before['source_frames_over_10mm'],'after_frames_over_10mm':after['source_frames_over_10mm'],'raw_reconstruction_preserved':True}
 c['training_reference']=reference;(r/'episode-config.json').write_text(json.dumps(c,indent=2));marker.write_text(json.dumps(out,indent=2));print(ep,out,flush=True);return out
def prepare(ep):
 r,c=context(ep)
 with (r/'research/collision-reference.lock').open('a') as lock:
  fcntl.flock(lock,fcntl.LOCK_EX)
  return _prepare(ep)
if __name__=='__main__':
 failed=[]
 for ep in map(int,sys.argv[1:]):
  try:prepare(ep)
  except Exception:
   import traceback
   r,c=context(ep);message=traceback.format_exc();(r/'research/contact-preparation-failure.json').write_text(json.dumps({'episode':ep,'traceback':message},indent=2));print('Contact preparation FAILED',ep,message,flush=True);failed.append(ep)
 if failed:raise SystemExit(1)
