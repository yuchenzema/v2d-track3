"""Preserve earlier batch policies and resumable states using filesystem hard links."""
from pathlib import Path
import hashlib,json,shlex,subprocess,time
BASE=Path('/home/yu_eli/v2d-track3')
LINK_CODE="""import os,sys
from pathlib import Path
source=Path('/episode/flash_chord_coherent_actor_transfer');destination=Path('/episode/research/checkpoint-archive');destination.mkdir(exist_ok=True)
for name in sys.argv[1:]:
 assert name.startswith(('policy_','state_')) and name.endswith('.safetensors') and '/' not in name
 try:os.link(source/name,destination/name)
 except (FileExistsError,FileNotFoundError):pass
"""
def scan():
 for r in (BASE/'runs').glob('episode_*_newton_batch_20261015'):
  folder=r/'flash_chord_coherent_actor_transfer'
  if not folder.exists():continue
  c=json.loads((r/'episode-config.json').read_text());archive=r/'research/checkpoint-archive';archive.mkdir(exist_ok=True);new=[]
  for policy in folder.glob('policy_*.safetensors'):
   state=policy.with_name(policy.name.replace('policy_','state_',1))
   if not state.exists() or min(time.time()-policy.stat().st_mtime,time.time()-state.stat().st_mtime)<3:continue
   new.extend(p.name for p in [policy,state] if not (archive/p.name).exists())
  if not new:continue
  command=['docker','exec',f'v2d-batch-episode{c["episode"]}-newton','python','-c',LINK_CODE,*new]
  subprocess.run(['sg','docker','-c',shlex.join(command)],check=True,capture_output=True,text=True)
  manifest=archive/'manifest.json';data=json.loads(manifest.read_text()) if manifest.exists() else {'purpose':__doc__,'files':{}}
  for name in new:
   p=archive/name
   if not p.exists():continue
   step=int(p.stem.split('_')[-1]);data['files'][name]={'bytes':p.stat().st_size,'sha256':hashlib.sha256(p.read_bytes()).hexdigest(),'environment_steps':step,'additional_environment_steps':step-(250003456 if len(c['objects'])==2 else 0)}
  tmp=manifest.with_suffix('.tmp');tmp.write_text(json.dumps(data,indent=2));tmp.replace(manifest);print(c['episode'],'preserved',new,flush=True)
if __name__=='__main__':
 while True:
  try:scan()
  except (FileNotFoundError,json.JSONDecodeError,subprocess.CalledProcessError) as error:print('Archive scan retry:',str(error),flush=True)
  time.sleep(5)
