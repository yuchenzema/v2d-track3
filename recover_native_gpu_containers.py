"""Restore GPU access using fresh canonical containers while retaining live learners by ID.

NVIDIA documents that systemd device-cgroup updates can revoke GPU access in
legacy --gpus containers. Explicit device declarations protect the replacement
containers. Renaming the old container keeps its processes and open GPU contexts.
"""
from pathlib import Path
import fcntl,json,os,shlex,stat,subprocess,sys,time
from batch_worker import BASE,PRIOR,context
from batch_native import _docker,active_learners

def probe(name):
 command=['sg','docker','-c',shlex.join(['docker','exec',name,'nvidia-smi','--query-gpu=uuid','--format=csv,noheader'])]
 return subprocess.run(command,capture_output=True,text=True)

with (BASE/'runs/evaluation_batch_20261015/gpu-container-recovery.lock').open('a') as lock:
 fcntl.flock(lock,fcntl.LOCK_EX)
 for ep in map(int,sys.argv[1:]):
  r,c=context(ep);name=f'v2d-batch-episode{ep}-newton';result=probe(name)
  if result.returncode==0:print(ep,'GPU already available',flush=True);continue
  before=active_learners();stamp=time.time_ns();staging=f'{name}-recovery-{stamp}';legacy=f'{name}-legacy-{stamp}';template=json.loads((PRIOR/'research/container-commands.json').read_text())[0][1:];command=[s.replace(str(PRIOR),str(r)).replace('v2d-episode5-newton',staging) for s in template];image=_docker(['inspect','--format','{{.Image}}',name]).stdout.strip();command=[image if s=='flash_chord' else s for s in command];nodes=[p for p in [*Path('/dev').glob('nvidia*'),*Path('/dev/nvidia-caps').glob('*')] if stat.S_ISCHR(p.stat().st_mode)];extra=['--runtime','nvidia']
  for node in sorted(nodes):extra+=['--device',str(node)]
  command[1:1]=extra;_docker(command,r/'logs/create-recovered-gpu-container.log');fresh=probe(staging)
  if fresh.returncode:raise RuntimeError(f'Replacement container also lacks GPU access: {fresh.stdout} {fresh.stderr}')
  _docker(['rename',name,legacy]);_docker(['rename',staging,name]);after=active_learners();assert set(before)<=set(after),'A pre-existing learner disappeared during a name-only container replacement'
  report={'method':__doc__,'original_container_retained':legacy,'canonical_replacement':name,'exact_original_image_id':image,'explicit_devices':[str(x) for x in sorted(nodes)],'gpu_uuid_probe':fresh.stdout.strip(),'learners_before':before,'learners_after':after,'live_learning_processes_not_restarted':True,'official_troubleshooting':'https://docs.nvidia.com/datacenter/cloud-native/container-toolkit/latest/troubleshooting.html#containers-losing-access-to-gpus-with-error-failed-to-initialize-nvml-unknown-error'};path=r/'research/gpu-container-recovery.json';history=json.loads(path.read_text()) if path.exists() else [];history.append(report);path.write_text(json.dumps(history,indent=2));print(ep,'GPU access recovered; learners retained',after,flush=True)
