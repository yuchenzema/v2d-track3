"""Per-episode Newton/retarget container runner with durable command logs."""
from pathlib import Path
import json,shlex,subprocess,sys,time,fcntl,hashlib,re,shutil,os
BASE=Path('/home/yu_eli/v2d-track3');PRIOR=BASE/'runs/episode_000005_newton_20261015'

def active_learners():
 found=[]
 for path in Path('/proc').glob('[0-9]*/cmdline'):
  try:
   argv=path.read_bytes().split(b'\0')
   if argv and Path(os.fsdecode(argv[0])).name.startswith('python') and b'/episode/scripts/train_coherent_actor_transfer.py' in argv:found.append(int(path.parent.name))
  except (OSError,ValueError):pass
 return found

def training_priority_allows(root,cfg):
 path=BASE/'runs/evaluation_batch_20261015/training-priority.json'
 if not path.exists():return True
 priority=json.loads(path.read_text())
 return (not priority.get('active') or root.name in priority.get('allowed_run_names',[])
         or cfg.get('episode') in priority.get('preserve_running_episodes',[]))

def training_capacity(root,required,existing_count):
 """One learner only, retaining GPU and host-memory headroom."""
 maximum=1
 minimum=required+4000
 available={line.split(':')[0]:int(line.split()[1])*1024 for line in Path('/proc/meminfo').read_text().splitlines()}
 memory_ok=available['MemAvailable']>=12*1024**3
 return maximum,minimum,memory_ok

def launch_training(container,root):
 """Serialize memory-bounded admission, including learners from older queues."""
 source=(root/'scripts/train_coherent_actor_transfer.py').read_text();cfg=json.loads((root/'episode-config.json').read_text());worlds=int(cfg['training_world_count']) if 'training_world_count' in cfg else int(re.search(r'scene.world_count=(\d+)',source)[1]);required=10500 if worlds>=2048 else (7500 if len(cfg['objects'])==2 else 6000)
 log=root/'logs/newton-training.log';command=['sg','docker','-c',shlex.join(['docker','exec',container,'python','-u','/episode/scripts/train_coherent_actor_transfer.py'])]
 with Path('/tmp/v2d-track3-training-admission.lock').open('a') as admission:
  while True:
   fcntl.flock(admission,fcntl.LOCK_EX)
   existing=active_learners();free=int(subprocess.check_output(['nvidia-smi','--query-gpu=memory.free','--format=csv,noheader,nounits'],text=True).splitlines()[0])
   maximum,minimum,memory_ok=training_capacity(root,required,len(existing))
   disk=os.statvfs(BASE);free_disk=disk.f_bavail*disk.f_frsize
   if training_priority_allows(root,cfg) and len(existing)<maximum and free>=minimum and memory_ok and free_disk>=6_000_000_000:break
   # Waiting jobs release admission so a newly prioritized episode can proceed.
   fcntl.flock(admission,fcntl.LOCK_UN)
   time.sleep(15)
  (root/'research/training-admission.json').write_text(json.dumps({'existing_gpu_learner_pids':existing,'maximum_concurrent_learners':maximum,'free_gpu_memory_mb_at_admission':free,'minimum_free_gpu_memory_mb':minimum,'evaluation_memory_reserve_mb':minimum-required,'world_count':worlds,'utc':time.strftime('%Y-%m-%dT%H:%M:%SZ',time.gmtime())},indent=2));log.with_suffix('.command.json').write_text(json.dumps(command,indent=2))
  output=log.open('w');process=subprocess.Popen(command,stdout=output,stderr=subprocess.STDOUT)
  # Keep admission serialized until the container's Python process is visible.
  while process.poll() is None and not(set(active_learners())-set(existing)):time.sleep(.2)
  from episode_execution_guard import register_episode_child
  for pid in set(active_learners())-set(existing):register_episode_child(pid)
  fcntl.flock(admission,fcntl.LOCK_UN)
 try:code=process.wait()
 finally:output.close()
 if code:raise subprocess.CalledProcessError(code,command)
def _docker(args,log=None):
 command=['sg','docker','-c',shlex.join(['docker',*args])]
 if log:
  log.with_suffix('.command.json').write_text(json.dumps(command,indent=2))
  with log.open('w') as f:return subprocess.run(command,stdout=f,stderr=subprocess.STDOUT,check=True)
 return subprocess.run(command,check=True,capture_output=True,text=True)
def docker(args,log=None):
 if any(s.endswith(('/native_reference_audit.py','/optimize_native_grasps.py','/project_native_object_contacts.py','/refine_native_object_poses.py')) for s in args):
  assert log is not None
  root=log.parent.parent;assert root.name.endswith('_newton_batch_20261015')
  shutil.copy2(BASE/'run_geometry_cpu.py',root/'scripts/run_geometry_cpu.py')
  args=list(args);index=args.index('python');assert args[index+1]=='-u';target=args[index+2]
  args[index:index+3]=['env','CUDA_VISIBLE_DEVICES=','JAX_PLATFORMS=cpu','python','-u','/episode/scripts/run_geometry_cpu.py',target]
  return _docker(args,log)
 if args and args[0]=='exec' and '/isaac-sim/python.sh' not in args and not any('train_coherent_actor_transfer.py' in s for s in args):
  from batch_gpu_gate import auxiliary_gpu
  small='evaluation.world_count=32' in args or any(s.endswith(('/record_policy_states.py','/record_velocity_controller.py','/record_hybrid_controller.py','/diagnose_zero_residual.py','/diagnose_terminal_support.py','/render_source_camera.py','/render_recorded_oblique.py','/record_reconstruction.py')) for s in args)
  minimum=2200 if small else 3500
  if 'evaluation.world_count=128' in args and log:
   config_path=next(parent/'episode-config.json' for parent in log.parents if (parent/'episode-config.json').is_file())
   config=json.loads(config_path.read_text());minimum=2200 if len(config['objects'])==1 else 2700
  with auxiliary_gpu('native',minimum_free_mb=minimum):return _docker(args,log)
 return _docker(args,log)
def containers(root,ep):
 (root/'human_motion_data').mkdir(exist_ok=True)
 commands=json.loads((PRIOR/'research/container-commands.json').read_text());names=[]
 for label,command in zip(['newton','retarget'],commands):
  name=f'v2d-batch-episode{ep}-{label}';names.append(name)
  exists=subprocess.run(['sg','docker','-c',shlex.join(['docker','inspect',name])],stdout=subprocess.DEVNULL,stderr=subprocess.DEVNULL).returncode==0
  if not exists:
   command=[s.replace(str(PRIOR),str(root)).replace(f'v2d-episode5-{label}',name) for s in command[1:]]
   if label=='newton':
    import stat
    extra=['--runtime','nvidia']
    for node in sorted([*Path('/dev').glob('nvidia*'),*Path('/dev/nvidia-caps').glob('*')]):
     if stat.S_ISCHR(node.stat().st_mode):extra+=['--device',str(node)]
    command[1:1]=extra
   docker(command,root/f'logs/create-{label}-container.log')
 return names
if __name__=='__main__':
 ep=int(sys.argv[1]);stage=sys.argv[2];root=BASE/'runs'/f'episode_{ep:06d}_newton_batch_20261015';newton,retarget=containers(root,ep)
 if stage=='retarget':
  while not (root/'reconstruction/scene_motion_physical.npz').exists():time.sleep(5)
  for script in ['retarget_hands','coherent_grasp_variant']:
   docker(['exec',retarget,'/isaac-sim/python.sh','-u',f'/episode/scripts/{script}.py'],root/f'logs/{script}.log')
  print(ep,'retarget complete',flush=True)
 elif stage=='train':
  folder=root/'flash_chord_coherent_actor_transfer';folder.mkdir(exist_ok=True)
  with (folder/'training-process.lock').open('a') as lock:
   fcntl.flock(lock,fcntl.LOCK_EX)
   if not (folder/'TRAINING_COMPLETED').exists():
    while True:
     pending=json.loads((root/'episode-config.json').read_text()).get('reference_review_pending',False)
     if not pending or (root/'coherent-input-checksums.json').exists():break
     time.sleep(10)
    cfg=json.loads((root/'episode-config.json').read_text());substeps=int(cfg.get('physics_substeps',5));script=root/'scripts/train_coherent_actor_transfer.py';source=script.read_text()
    shutil.copy2(BASE/'validation_stop.py',root/'scripts/validation_stop.py')
    if 'maybe_validation_stop' not in source:
     source=source.replace('from pathlib import Path','from pathlib import Path\nfrom validation_stop import TrainingValidated,maybe_validation_stop\nVALIDATED_STOP_PROTOCOL=2',1)
     source=source.replace(' def close(self):self.file.close()','  maybe_validation_stop(ROOT,int(step),START)\n def close(self):self.file.close()',1)
     source=source.replace(' finally:sink.close()'," except TrainingValidated as decision:\n  print('Validated early stopping:',decision,flush=True)\n  (ROOT/'flash_chord_coherent_actor_transfer/TRAINING_COMPLETED').write_text('Validated early stopping; resumable checkpoint preserved.\\n')\n  raise SystemExit(0)\n finally:sink.close()",1)
    source=re.sub(r",'sim.substeps=[0-9]+'",'',source);source=source.replace("'task.motion_speed=0.5'",f"'task.motion_speed=0.5','sim.substeps={substeps}'",1)
    checksum=root/'coherent-input-checksums.json'
    if not checksum.exists():
     subprocess.run([str(PRIOR/'.venv-video/bin/python'),str(BASE/'trim_release_contact_targets.py'),'--run',str(root)],check=True)
     # Freeze actual URDF collision meshes and support geometry as well as the
     # image reconstruction. Existing learners keep their original snapshots.
     source=source.replace("*sorted((ROOT/'reconstruction/mesh_priors').glob('*.ply'))]","*sorted((ROOT/'reconstruction/mesh_priors').glob('*.ply')),*sorted((ROOT/'reconstruction/mesh_priors').glob('*.obj')),*sorted((Path(reference).parents[2]/'reconstructed_stage').glob('*.usda'))]")
    if checksum.exists():
     for path,digest in json.loads(checksum.read_text()).items():assert hashlib.sha256((root/path).read_bytes()).hexdigest()==digest,('Frozen training input changed',path)
     saved=json.loads((root/'coherent-training-config.json').read_text());assert int(saved['env']['sim']['substeps'])==substeps,'Use a separate run when changing an already trained physics model'
    states=sorted(folder.glob('state_*.safetensors'),key=lambda p:int(p.stem.split('_')[-1]));states=[p for p in states if p.with_name('policy_'+p.stem.split('_')[-1]+'.safetensors').exists()];resume=states[-1] if states else None
    if resume:
     prior_literal="'training.checkpoint_path=/prior/state_250003456.safetensors'";own=f"'training.checkpoint_path=/episode/flash_chord_coherent_actor_transfer/{resume.name}'"
     if prior_literal in source:source=source.replace(prior_literal,own).replace('training.checkpoint.load_optimizer=false','training.checkpoint.load_optimizer=true').replace('training.checkpoint.load_reward_normalizer=false','training.checkpoint.load_reward_normalizer=true')
     else:source=source.replace("'output_dir=/episode/flash_chord_coherent_actor_transfer'",f"'training.resume=true',{own},'training.checkpoint.load_optimizer=true','training.checkpoint.load_reward_normalizer=true','output_dir=/episode/flash_chord_coherent_actor_transfer'",1)
     history=root/'research/training-resume-sessions.json';rows=json.loads(history.read_text()) if history.exists() else [];rows.append({'checkpoint':str(resume.relative_to(root)),'sha256':hashlib.sha256(resume.read_bytes()).hexdigest(),'optimizer_and_critic_restored':True,'replay_restarted':True,'utc':time.strftime('%Y-%m-%dT%H:%M:%SZ',time.gmtime())});history.write_text(json.dumps(rows,indent=2))
    script.write_text(source)
    launch_training(newton,root)
  print(ep,'training complete',flush=True)
 else:raise ValueError(f'Unknown stage {stage!r}')
