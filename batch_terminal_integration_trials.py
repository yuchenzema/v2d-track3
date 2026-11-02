"""Test finer integration before accepting a larger geometric correction."""
import fcntl,json,shutil,sys
from batch_worker import context,BASE
from batch_native import containers,docker
for ep in map(int,sys.argv[1:]):
 r,c=context(ep)
 with (r/'research/collision-reference.lock').open('a') as lock:
  fcntl.flock(lock,fcntl.LOCK_EX);assert not (r/'coherent-input-checksums.json').exists();native,_=containers(r,ep);shutil.copy2(BASE/'diagnose_terminal_support.py',r/'scripts/diagnose_terminal_support.py');trials=[]
  for substeps in [20,40]:
   label=f'terminal-support-original-{20*substeps}hz';docker(['exec',native,'env',f'V2D_DIAGNOSTIC_SUBSTEPS={substeps}',f'V2D_DIAGNOSTIC_LABEL={label}','python','-u','/episode/scripts/diagnose_terminal_support.py'],r/f'logs/{label}.log');report=json.loads((r/f'evaluation/{label}.json').read_text());target=next(v for v in report['objects'] if v['object']==('blue_cup' if ep==14 else 'white_pot'));trials.append(report);print(ep,20*substeps,target,flush=True)
  (r/'research/terminal-support-integration-trials.json').write_text(json.dumps(trials,indent=2))
