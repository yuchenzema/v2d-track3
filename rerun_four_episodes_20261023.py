"""Cold reconstruction and independent policy training for four reviewed failures.
Every stage writes durable status; previous releases and their inputs are untouched.
"""
import os,json,shutil,stat,time,traceback,subprocess,fcntl,concurrent.futures,threading,sys
from pathlib import Path
from improvement_common import BASE,BATCH,save,utc,digest,better,verify_preserved,preserve
from batch_native import _docker,docker,launch_training
import batch_worker as vision
from batch_generic_retarget import install
from object_motion_quality import attach
from video_catalog import catalog
EPISODES=[19,20,30,33];OUT=BATCH/'improvements_20261023_rebuild';VARIANT='full_rebuild_20261023';PY=BASE/'runs/episode_000005_newton_20261015/.venv-video/bin/python'
def root(ep):return BASE/f'runs/episode_{ep:06d}_newton_{VARIANT}'
def status(ep,phase,state='running',**extra):
 r=root(ep);entry={'episode':ep,'pid':os.getpid(),'process_script':Path(sys.argv[0]).name,'updated_utc':utc(),'status':state,'phase':phase,'description':phase,'run':str(r),'activity_kind':'complete_reconstruction_and_retraining',**extra};save(r/'research/rebuild-status.json',entry);save(OUT/f'episode_{ep:06d}-status.json',entry)
 save(BATCH/f'improvements_20261021/activities/episode_{ep:06d}-full-rebuild-20261023.json',entry)
 phase_status=json.loads((OUT/'phase-status.json').read_text()) if (OUT/'phase-status.json').exists() else {'episodes':{}}
 phase_status['episodes'][str(ep)]=entry;phase_status.update(updated_utc=utc(),worker_pid=os.getpid(),requested_episodes=EPISODES);save(OUT/'phase-status.json',phase_status)
def command(r,args,label):vision.run([str(x) for x in args],r/f'logs/{label}.log')
def prepare(ep):
 r=root(ep);source=BASE/f'runs/episode_{ep:06d}_newton_batch_20261015'
 if (r/'episode-config.json').exists():return r,json.loads((r/'episode-config.json').read_text())
 r.mkdir();cfg=json.loads((source/'episode-config.json').read_text())
 for directory in ['scripts','research','logs','reconstruction','evaluation','videos','deliverables','human_motion_data','flash_chord_coherent_actor_transfer']:(r/directory).mkdir()
 (r/'.venv-video').symlink_to(PY.parent.parent,target_is_directory=True);(r/'models').symlink_to(source/'models',target_is_directory=True)
 for key in ['physical_grasp_refinement','predecomposed_object_collisions','training_seed_checkpoint_run','contact_full_state_seed','training_initial_counter','adaptive_single_prefix','wrist_side_motor_profiles']:
  cfg.pop(key,None)
 cfg.update(run=str(r),baseline_run=str(source),training_variant=VARIANT,training_reference='track3_collision_variant',native_container=f'v2d-rebuild23-episode{ep}-newton',training_initial_counter=0,training_budget=100_000_256,training_world_count=512,training_curriculum='fresh_rebuild',training_compute_dtype='float32',physics_substeps=20,physics_integration_hz=400,physics_nconmax=2048,physics_njmax=4096,requires_verified_improvement_for_publication=True,all_frames_visually_reviewed=False,all_source_frames_decoded_and_automatically_checked=True,reference_review_pending=True,improvement_experiment='Cold video reconstruction followed by independently initialized FlashSAC; audited contact and release targets',frame_cache_format='png',wrist_motor_profile={'kp':250.,'kd':20.})
 # Resting the spoon on the table and releasing the hand are different source events.
 if ep==33:cfg['source_support_intervals']={'wooden_spoon':[[0,56],[289,cfg['frames']-1]]}
 if ep in [19,30]:cfg['maximum_floor_projection_m']=.04
 cfg['training_focus']={'source_interval':[0,min(cfg['frames']-1,max(g['start'] for g in cfg['grasps'])+16)],'source_frame_count':cfg['frames'],'probability':.25}
 cfg['improvement_experiment']='Fresh source reconstruction, full-trajectory collision fitting, pickup-focused independent100M-step training'
 save(r/'episode-config.json',cfg)
 for name in ['sam2_prompts.json']:
  shutil.copy2(source/'reconstruction'/name,r/'reconstruction'/name)
 shutil.copy2(source/'research/source-video-review.json',r/'research/source-video-review.json')
 save(r/'research/rebuild-input-provenance.json',{'updated_utc':utc(),'source_videos':[{'path':str(BASE/f'datasets/video_to_data_challenge/track_3/evaluation/videos/chunk-000/observation.images.ego_cam_{cam}/episode_{ep:06d}.mp4'),'sha256':digest(BASE/f'datasets/video_to_data_challenge/track_3/evaluation/videos/chunk-000/observation.images.ego_cam_{cam}/episode_{ep:06d}.mp4')} for cam in 'abc'],'old_estimated_camera_object_hand_poses_copied':False,'old_masks_or_depth_copied':False,'old_policy_or_optimizer_copied':False,'shared_downloaded_model_weights_reused':True,'source_event_annotations_reused':True,'episode33_spoon_supported_on_table_from_source_frame':289 if ep==33 else None,'evaluation_ground_truth_used':False})
 save(r/'research/improvement-status.json',{'episode':ep,'status':'reconstructing','updated_utc':utc(),'run':str(r)})
 status(ep,'Queued for complete source-video reconstruction',state='queued');return r,cfg

def make_container(r,ep,kind):
 name=f'v2d-rebuild23-episode{ep}-{kind}'
 existing=subprocess.run(['sg','docker','-c',f'docker inspect {name}'],capture_output=True,text=True)
 if existing.returncode==0:
  if not json.loads(existing.stdout)[0]['State']['Running']:_docker(['start',name],r/f'logs/restart-{kind}.log')
  return name
 info=json.loads(_docker(['inspect',f'v2d-batch-episode{ep}-{kind}']).stdout)[0];args=['run','-d','--gpus','all','--runtime','nvidia','--ipc','host','--name',name]
 for mount in info['Mounts']:
  args+=['-v',(str(r) if mount['Destination']=='/episode' else mount['Source'])+':'+mount['Destination']+('' if mount['RW'] else ':ro')]
 for device in sorted([*Path('/dev').glob('nvidia*'),*Path('/dev/nvidia-caps').glob('*')]):
  if stat.S_ISCHR(device.stat().st_mode):args+=['--device',str(device)]
 args+=['-e','PYGLET_HEADLESS=true','-e','XLA_PYTHON_CLIENT_PREALLOCATE=false']
 if kind=='newton':args+=['--workdir','/workspace']
 args+=['--entrypoint','/bin/sleep',info['Image'],'infinity'];_docker(args,r/f'logs/create-{kind}.log');return name

def guarded_vision(args,log):
 from episode_execution_guard import available_ram_bytes
 while True:
  free=int(subprocess.check_output(['nvidia-smi','--query-gpu=memory.free','--format=csv,noheader,nounits'],text=True).splitlines()[0])
  if free>=16500 and available_ram_bytes()>=12*1024**3:break
  time.sleep(15)
 return ORIGINAL_VISION_DOCKER(args,log)
ORIGINAL_VISION_DOCKER=vision.docker;vision.docker=guarded_vision

def rebuild(ep):
 r,cfg=prepare(ep);done=r/'research/full-reconstruction-completed.json'
 if done.exists():return
 stages=[('calibrate_and_measure_hands',lambda:vision.prepare(r,cfg)),('fresh_segmentation_depth_and_stereo',lambda:vision.metric(r,cfg)),('fresh_slam_foundationpose_and_source_fit',lambda:vision.generic_reconstruction(r,cfg))]
 for name,action in stages:
  marker=r/f'research/rebuild-{name}-completed.json'
  if marker.exists():continue
  status(ep,name.replace('_',' '));action();save(marker,{'completed_utc':utc()})
 status(ep,'Reconstructing tool landmarks and correcting source-supported object motion')
 command(r,[PY,BASE/'rebuild_source_constraints_20261022.py','--run',r],'rebuild-source-constraints')
 install(r)
 if ep==33:command(r,[PY,BASE/'repair_supported_spoon_20261022.py','--run',r],'rebuild-supported-spoon')
 command(r,[PY,BASE/'review_reconstructed_source_20261023.py','--run',r],'rebuild-source-review')
 command(r,[PY,r/'scripts/assess_source_fidelity.py','--output','reference-source-fidelity','--overlay'],'rebuild-reference-fidelity')
 if not (r/'reconstruction/rebuilt_scene.glb').exists():command(r,[PY,BASE/'batch_visuals.py',r],'rebuild-visuals')
 native=make_container(r,ep,'newton');retarget=make_container(r,ep,'retarget')
 for helper in ['native_reference_audit.py','rigid_pose_collision_audit.py','optimize_native_grasps.py','native_pinch_contacts.py','grasp_physics_checks.py','apply_collision_grasps.py','apply_collision_transitions.py','project_native_object_contacts.py','evaluate_reference_controller.py','run_geometry_cpu.py','robot_wrist_calibration.py','training_numeric_failure_guard.py','training_finger_reset.py','validation_stop.py']:
  shutil.copy2(BASE/helper,r/'scripts'/helper)
 for script in ['retarget_hands','coherent_grasp_variant']:
  if not (r/f'research/rebuild-{script}-completed.json').exists():
   status(ep,'Rebuilding robot grasps: '+script);docker(['exec',retarget,'/isaac-sim/python.sh','-u',f'/episode/scripts/{script}.py'],r/f'logs/rebuild-{script}.log');save(r/f'research/rebuild-{script}-completed.json',{'completed_utc':utc()})
 def native_cpu(script,label,**environment):
  _docker(['exec',native,'env','CUDA_VISIBLE_DEVICES=','JAX_PLATFORMS=cpu',*[f'{k}={v}' for k,v in environment.items()],'python','-u','/episode/scripts/run_geometry_cpu.py',f'/episode/scripts/{script}.py'],r/f'logs/{label}.log')
 status(ep,'Fitting robot grasps against native collision geometry')
 if not (r/'research/collision-aware-applied-grasps.json').exists():
  native_cpu('optimize_native_grasps','rebuild-collision-grasp-fit')
  docker(['exec',retarget,'/isaac-sim/python.sh','-u','/episode/scripts/apply_collision_grasps.py'],r/'logs/rebuild-grasp-apply.log')
 status(ep,'Rebuilding open-finger release and collision-safe withdrawal targets')
 command(r,[PY,BASE/'rebuild_release_targets_20261022.py','--run',r],'rebuild-release-targets')
 docker(['exec',retarget,'env','V2D_TRANSITION_REFERENCE=track3_collision_variant','V2D_REBUILD_ONLY=1','V2D_CONTACT_APPLY_OUTPUT=research/rebuild-release-fk.json','/isaac-sim/python.sh','-u','/episode/scripts/apply_collision_transitions.py'],r/'logs/rebuild-release-fk.log')
 command(r,[PY,BASE/'trim_release_contact_targets.py','--run',r],'rebuild-release-contact-labels')
 if cfg.get('approved_source_cad_run'):
  # Detect a pose-changing legacy preflight before expensive full-contact fits.
  from source_cad_training_gate import require_source_cad_alignment
  require_source_cad_alignment(ep,r)
 native_cpu('native_reference_audit','rebuild-final-collision-audit',V2D_AUDIT_REFERENCE='track3_collision_variant')
 status(ep,'Fitting hand contacts across every reconstructed source frame')
 command(r,[PY,BASE/'fit_full_trajectory_contacts_20261023.py','--run',r],'full-trajectory-contact-fit')
 cfg=json.loads((r/'episode-config.json').read_text());cfg['reference_review_pending']=False;save(r/'episode-config.json',cfg)
 save(done,{'completed_utc':utc(),'all_vision_stages_rerun':True,'source_frames':cfg['frames'],'retarget_rebuilt':True,'old_policy_reused':False,'reference':'track3_collision_variant','collision_report':'evaluation/track3_collision_variant-collision-audit.json'})
 status(ep,'Fresh reconstruction and retargeting completed; queued for independent training',state='queued',training_started=False)

def train_and_export(ep):
 from source_cad_training_gate import require_source_cad_alignment
 require_source_cad_alignment(ep,root(ep))
 from batch_review import monitor
 from batch_export import export
 from batch_publish_runtime import publish
 from batch_final_audit import audit
 r=root(ep);cfg=json.loads((r/'episode-config.json').read_text());native=cfg['native_container'];script=r/'scripts/train_coherent_actor_transfer.py'
 if cfg.get('require_nonpenetrating_reference_before_training') and not (r/'coherent-input-checksums.json').exists():
  for helper in ['native_reference_audit.py','rigid_pose_collision_audit.py','optimize_native_grasps.py','native_pinch_contacts.py','grasp_physics_checks.py']:
   shutil.copy2(BASE/helper,r/'scripts'/helper)
  command(r,[PY,BASE/'fit_full_trajectory_contacts_20261023.py','--run',r],'pretraining-nonpenetration-fit')
  cfg=json.loads((r/'episode-config.json').read_text())
 if cfg.get('validated_video_pipeline'):
  gate=r/'research/approach-grasp-probes.json'
  reference=r/f"human_motion_data/{cfg['training_reference']}/processed/sequence_id={cfg['sequence']}/robot_name=sharpa_wave/motion.parquet"
  cached=json.loads(gate.read_text()) if gate.exists() else {}
  if cached.get('reference_parquet_sha256')!=digest(reference):
   for name in ['approach_grasp_probe.py','grasp_physics_checks.py','legacy_source_frame_playback.py']:shutil.copy2(BASE/name,r/'scripts'/name)
   status(ep,'Unassisted grasp tests from before approach, including model friction-cone support checks')
   docker(['exec',native,'python','-u','/episode/scripts/approach_grasp_probe.py'],r/'logs/approach-grasp-probes.log')
  from grasp_training_admission import decide
  policy_path=BATCH/'focused_episodes_20261028/experimental-training-policy.json'
  experiment_policy=json.loads(policy_path.read_text()) if policy_path.exists() else {}
  grasp_admission=decide(cfg,experiment_policy,json.loads(gate.read_text()),digest(reference))
  save(r/'research/grasp-training-admission.json',dict(grasp_admission,updated_utc=utc()))
  cfg.update(pretraining_grasp_passed=grasp_admission['pretraining_grasp_passed'],
             post_training_grasp_acceptance_required=True)
  save(r/'episode-config.json',cfg)
 if not (r/'coherent-input-checksums.json').exists():
  shutil.copy2(BASE/'legacy_source_frame_playback.py',r/'scripts/legacy_source_frame_playback.py')
  s=(BASE/'fresh_improvement_training.py').read_text().replace('BUDGET = 50_000_896',"BUDGET = int(episode['training_budget'])").replace('scene.world_count=1024','scene.world_count=512').replace('training.compute_dtype=bfloat16','training.compute_dtype=float32').replace('training.checkpoint.save_interval_environment_steps=10000000','training.checkpoint.save_interval_environment_steps=10000000').replace("'sim.njmax=1536'","'sim.nconmax=2048','sim.njmax=4096'").replace("'reset.seed=43'",f"'reset.seed={20261023+ep}'")
  s=s.replace("    first = OmegaConf.to_container(cfg.curriculum.stages[0], resolve=True)","    from flash_chord.embodiments.sharpa_hands import SharpaHandsConfig\n    from omegaconf import open_dict\n    overrides=dict(SharpaHandsConfig().joint_overrides)\n    overrides[r'wrist_[xyz]$']={**overrides[r'wrist_[xyz]$'],**episode['wrist_motor_profile']}\n    with open_dict(cfg.embodiment.config):cfg.embodiment.config.joint_overrides=overrides\n    first = OmegaConf.to_container(cfg.curriculum.stages[0], resolve=True)")
  s=s.replace("'baseline_inputs_copied_without_pose_or_geometry_edits': True","'baseline_inputs_copied_without_pose_or_geometry_edits': False, 'reconstruction_rerun_from_original_video': True")
  s=s.replace("'world_count': 1024","'world_count': 512").replace("'seed': 43",f"'seed': {20261023+ep}")
  s=s.replace("install_training_fingers(native['main'].__wrapped__.__globals__)","install_training_fingers(native['main'].__wrapped__.__globals__,focus=episode.get('training_focus'))")
  script.write_text(s)
  _docker(['exec',native,'env','CUDA_VISIBLE_DEVICES=','JAX_PLATFORMS=cpu','V2D_VALIDATE_CONFIG_ONLY=1','python','-u','/episode/scripts/run_geometry_cpu.py','/episode/scripts/train_coherent_actor_transfer.py'],r/'logs/rebuild-training-config-validation.log')
 if not (r/'flash_chord_coherent_actor_transfer/TRAINING_COMPLETED').exists():
  status(ep,'Independent fresh FlashSAC policy training',state='running');save(r/'research/improvement-status.json',{'episode':ep,'status':'training','updated_utc':utc()})
  failed=threading.Event()
  def train():
   try:launch_training(native,r)
   except BaseException:failed.set();raise
  if cfg.get('validated_video_pipeline'):
   train();monitor(ep,root_override=r,container_override=native)
  else:
   with concurrent.futures.ThreadPoolExecutor(max_workers=2) as pool:
    trainer=pool.submit(train);reviewer=pool.submit(monitor,ep,root_override=r,container_override=native,stop_event=failed);trainer.result();reviewer.result()
 else:
  # A reviewer can fail while the learner finishes successfully (for example
  # GPU autotuning OOM). Recover all missing checkpoint reviews before export.
  status(ep,'Completing checkpoint validation after finished policy training')
  monitor(ep,root_override=r,container_override=native)
 status(ep,'Rendering and independently validating full-source policy rollout')
 if cfg.get('official_evaluation_required'):
  from official_policy_suite import evaluate_candidate
  suites=[]
  for checkpoint in sorted((r/'flash_chord_coherent_actor_transfer').glob('policy_*.safetensors')):
   status(ep,'Official evaluator: 16 worlds, six repeat trials and one recorded policy trajectory')
   suites.append(evaluate_candidate(r,checkpoint,native))
  assert suites,'No trained policy checkpoint available to official evaluator'
  save(r/'evaluation/official-candidate-suites.json',{'updated_utc':utc(),'candidates':suites,'strict_source_motion_verified':False,'submission_ready':False})
 save(r/'research/improvement-status.json',{'episode':ep,'status':'rendering','updated_utc':utc()})
 if not (r/'evaluation/reference-source-fidelity.json').exists():
  command(r,[PY,r/'scripts/assess_source_fidelity.py','--output','reference-source-fidelity'],'export-reference-fidelity')
 export(ep,root_override=r,container_override=native)
 if cfg.get('official_evaluation_required'):
  from official_policy_suite import bind_rendered_attempt
  bind_rendered_attempt(r)
 while time.time()-(r/'result-report.json').stat().st_mtime<11:time.sleep(1)
 # Diagnostic export completion is independent of manual source review and
 # motion acceptance. Keep every artifact/protocol check and report review
 # scope explicitly; video_catalog separately enforces publication gates.
 publish(r);validation=audit(r,require_source_review=False);assert validation['integrity_passed'],validation
 selected=json.loads((r/'final-controller-selection.json').read_text())['selected'];prior=catalog(ep,cfg['baseline_run'],True)['best'];previous=attach(prior['run'],prior['selected']);larger=json.loads((r/'evaluation/final-metrics.json').read_text())['metrics']['chord_sr'];improved=better(selected,previous) and larger>=prior['quality']['completion_rate_against_video_estimate']-.0625
 result={'episode':ep,'status':'completed','updated_utc':utc(),'run':str(r),'experiment':'Complete source-video reconstruction and independent fresh100M-step FlashSAC retraining','measured_improvement':improved,'baseline':previous,'candidate':selected,'artifact_integrity':validation,'full_human_motion_match_verified':False};save(OUT/f'episode_{ep:06d}-fresh-result.json',result);save(r/'research/improvement-status.json',result);verify_preserved(ep);command(r,[PY,BASE/'audit_rebuilt_rollout_20261023.py','--run',r],'all-source-frame-rollout-audit');status(ep,'New rollout rendered and checked; '+('verified improvement accepted' if improved else 'candidate retained for review; previous accepted video preserved'),state='completed',measured_improvement=improved)

def main():
 OUT.mkdir(exist_ok=True);failures={};save(OUT/'phase-status.json',{'phase':'full_reconstruction_and_training','updated_utc':utc(),'episodes':{},'requested_episodes':EPISODES})
 for ep in EPISODES:
  preserve(ep);prepare(ep);status(ep,'Queued for complete new reconstruction from original videos',state='queued')
 registry=BATCH/'improvements_20261021/active-runs.json';active=json.loads(registry.read_text()) if registry.exists() else {};active.update({str(ep):root(ep).name for ep in EPISODES});save(registry,active)
 for ep in EPISODES:
  try:rebuild(ep)
  except Exception:failures[str(ep)]=traceback.format_exc();status(ep,'Fresh reconstruction failed; retry required',state='failed',error=failures[str(ep)]);print(failures[str(ep)],flush=True)
 for ep in EPISODES:
  if str(ep) in failures:continue
  try:
   for attempt in range(3):
    try:train_and_export(ep);break
    except subprocess.CalledProcessError:
     if attempt==2:raise
     status(ep,'Retrying checkpoint validation/export with saved training preserved');time.sleep(20)
  except Exception:failures[str(ep)]=traceback.format_exc();status(ep,'Training/export failed; previous releases preserved',state='failed',error=failures[str(ep)]);save(root(ep)/'research/improvement-status.json',{'episode':ep,'status':'failed','updated_utc':utc(),'error':failures[str(ep)]});print(failures[str(ep)],flush=True)
 save(OUT/'phase-status.json',{'phase':'completed_with_experiment_failures' if failures else 'completed','updated_utc':utc(),'requested_episodes':EPISODES,'failures':failures})
 save(OUT/'worker-completion.json',{'updated_utc':utc(),'failures':failures,'requested':EPISODES,'all_four_finished':not failures})
if __name__=='__main__':
 OUT.mkdir(exist_ok=True)
 with (OUT/'worker.lock').open('a') as lock:
  fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB);main()
