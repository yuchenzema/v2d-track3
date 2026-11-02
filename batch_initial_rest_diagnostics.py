"""Run labeled unassisted PD diagnostics of source-observed rests, never label them policies."""
import fcntl,json,shutil,sys,time,subprocess
from batch_worker import context,BASE,PY
from batch_native import containers,docker
for ep in map(int,sys.argv[1:]):
 r,c=context(ep)
 while True:
  if ep in [14,15,16] and not (r/f'research/{c.get("training_reference","track3_collision_variant")}-initial-support-selection.json').exists():time.sleep(10);continue
  marker=r/'research/collision-reference-selection.json'
  if not marker.exists() or not json.loads(marker.read_text()).get('contact_refinement_complete'):time.sleep(10);continue
  break
 with (r/'research/collision-reference.lock').open('a') as lock:
  fcntl.flock(lock,fcntl.LOCK_EX);c=json.loads((r/'episode-config.json').read_text());newton,_=containers(r,ep)
  label='passive-rest-preflight';shutil.copy2(BASE/'diagnose_zero_residual.py',r/'scripts/diagnose_zero_residual.py')
  docker(['exec',newton,'env',f'V2D_DIAGNOSTIC_SUBSTEPS={c.get("physics_substeps",5)}',f'V2D_DIAGNOSTIC_LABEL={label}','python','-u','/episode/scripts/diagnose_zero_residual.py'],r/f'logs/{label}.log')
  subprocess.run([PY,str(BASE/'assess_passive_rests.py'),'--run',str(r),'--label',label],check=True)
  result=json.loads((r/f'evaluation/{label}-initial-rest.json').read_text());print(ep,result,flush=True)
