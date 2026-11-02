"""Serialize source-aware gravity searches without changing training references."""
import fcntl,shutil,sys,json,os
from batch_worker import context,BASE
from batch_native import containers,_docker
from batch_gpu_gate import auxiliary_gpu
for ep in map(int,sys.argv[1:]):
 r,c=context(ep)
 with (r/'research/collision-reference.lock').open('a') as lock:
  fcntl.flock(lock,fcntl.LOCK_EX);assert not (r/'coherent-input-checksums.json').exists()
  marker=r/'research/terminal-support-reference-selection.json'
  if os.environ.get('V2D_SKIP_ACCEPTED_SUPPORT')=='1' and marker.exists() and json.loads(marker.read_text()).get('accepted'):print(ep,'Keeping accepted supported placement',flush=True);continue
  native,_=containers(r,ep);shutil.copy2(BASE/'search_terminal_support.py',r/'scripts/search_terminal_support.py')
  override=['env',f"V2D_SUPPORT_OBJECT={int(os.environ['V2D_SUPPORT_OBJECT'])}"] if 'V2D_SUPPORT_OBJECT' in os.environ else []
  with auxiliary_gpu('native',minimum_free_mb=2200):_docker(['exec',native,*override,'python','-u','/episode/scripts/search_terminal_support.py'],r/'logs/terminal-support-search.log')
  print(ep,json.loads((r/'research/terminal-support-search.json').read_text())['best'],flush=True)
