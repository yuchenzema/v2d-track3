"""Prepare an isolated stable-contact cup learner from a finite preserved actor."""
import hashlib,json,shutil,stat
from pathlib import Path
from improvement_common import BASE,save,utc
from batch_native import _docker
OLD=BASE/'runs/episode_000010_newton_grasp_curriculum_20261020';R=BASE/'runs/episode_000010_newton_stable_contact_20261020';STEP=360_000_000
assert not (R/'coherent-input-checksums.json').exists() and not (R/'research/preparation-status.json').exists()
for p,h in json.loads((OLD/'coherent-input-checksums.json').read_text()).items():assert hashlib.sha256((OLD/p).read_bytes()).hexdigest()==h,p
R.mkdir(exist_ok=True)
for name in ['logs','research','evaluation','videos','deliverables','flash_chord_coherent_actor_transfer']:(R/name).mkdir(exist_ok=True)
for name in ['reconstruction','scripts']:shutil.copytree(OLD/name,R/name,dirs_exist_ok=True)
for name in ['track3_hollow_supported_candidate','track3_collision_variant','track3_video']:shutil.copytree(OLD/'human_motion_data'/name,R/'human_motion_data'/name,dirs_exist_ok=True)
if not (R/'.venv-video').exists():(R/'.venv-video').symlink_to(BASE/'runs/episode_000005_newton_20261015/.venv-video',target_is_directory=True)
cfg=json.loads((OLD/'episode-config.json').read_text());cfg.update(run=str(R),native_container='v2d-stable-episode10-newton',
 training_initial_counter=STEP,contact_training_additional_budget=40_008_192,training_budget=40_008_192,
 training_compute_dtype='float32',training_numeric_failure_guard=True,physics_substeps=20,
 training_variant='stable_contact_20261020',
 improvement_experiment='Finer contact simulation and fresh finite critic/optimizer around the preserved cup actor')
cfg.pop('training_seed_checkpoint_run',None)
save(R/'episode-config.json',cfg)
initial=R/'research/initial-policy';initial.mkdir(exist_ok=True);hashes={}
for kind in ['policy','state']:
 p=OLD/f'research/precision-schedule-checkpoint-{STEP}/{kind}_{STEP}.safetensors';target=initial/p.name
 shutil.copy2(p,target);hashes[p.name]=hashlib.sha256(p.read_bytes()).hexdigest();assert hashes[p.name]==hashlib.sha256(target.read_bytes()).hexdigest()
for name in ['source-human-video-recheck-20261020.json','source-video-review.json']:
 if (OLD/'research'/name).exists():shutil.copy2(OLD/'research'/name,R/'research'/name)
for name in ['reference-source-fidelity.json','physical-reference-source-fidelity.json','hollow-supported-reference-source-fidelity.json','hollow-supported-terminal-rest.json']:
 if (OLD/'evaluation'/name).exists():shutil.copy2(OLD/'evaluation'/name,R/'evaluation'/name)
selection=json.loads((OLD/'research/physical-grasp-reference-selection.json').read_text())
selection.update(updated_utc=utc(),training_initial_counter=STEP,verified_policy_improvement=False,physics_substeps=20,
 remaining_problem='Prior mixed-precision learner stopped on nonfinite actor loss. Preserve its data and checkpoints; use fresh learner state, finer physics and explicit failed-transition handling.')
save(R/'research/physical-grasp-reference-selection.json',selection)
save(R/'research/contact-finetune-adjustments.json',{'updated_utc':utc(),'source_interval':[48,135],
 'residual_mean_learning_rate_multiplier':.1,'training_translation_precision':{'fraction':.4,'scale':.02},
 'reason':'Train centimetre-scale cup placement with a fresh reward normalizer and critic; retain native orientation reward and canonical unassisted evaluation.'})
for source,target in [('contact_finetune_training.py','train_coherent_actor_transfer.py'),('training_finger_reset.py','training_finger_reset.py'),
 ('translation_precision_reward.py','translation_precision_reward.py'),('training_numeric_failure_guard.py','training_numeric_failure_guard.py'),
 ('validation_stop.py','validation_stop.py')]:shutil.copy2(BASE/source,R/'scripts'/target)
info=json.loads(_docker(['inspect','v2d-grasp-episode10-newton']).stdout)[0]
args=['run','-d','--gpus','all','--runtime','nvidia','--ipc','host','--name',cfg['native_container']]
for mount in info['Mounts']:
 source=str(R) if mount['Destination']=='/episode' else mount['Source']
 args+=['-v',source+':'+mount['Destination']+('' if mount['RW'] else ':ro')]
for node in sorted([*Path('/dev').glob('nvidia*'),*Path('/dev/nvidia-caps').glob('*')]):
 if stat.S_ISCHR(node.stat().st_mode):args+=['--device',str(node)]
args+=['-e','PYGLET_HEADLESS=true','-e','XLA_PYTHON_CLIENT_PREALLOCATE=false','--workdir','/workspace','--entrypoint','/bin/sleep',info['Image'],'infinity']
_docker(args,R/'logs/create-newton.log')
_docker(['exec',cfg['native_container'],'env','CUDA_VISIBLE_DEVICES=','JAX_PLATFORMS=cpu','V2D_VALIDATE_CONFIG_ONLY=1',
 'python','-u','/episode/scripts/train_coherent_actor_transfer.py'],R/'logs/stable-config-validation.log')
# Validate the actual transfer and guard wiring on CPU before allocating a learner.
_docker(['exec',cfg['native_container'],'env','CUDA_VISIBLE_DEVICES=','JAX_PLATFORMS=cpu','V2D_VALIDATE_ACTOR_TRANSFER_ONLY=1',
 'python','-u','/episode/scripts/run_geometry_cpu.py','/episode/scripts/train_coherent_actor_transfer.py'],R/'logs/stable-actor-transfer-validation.log')
save(R/'research/preparation-status.json',{'updated_utc':utc(),'status':'prepared_and_configuration_validated',
 'source_run':str(OLD),'actor_seed_environment_steps':STEP,'seed_checkpoint_sha256':hashes,'remaining_training_steps':40_008_192,
 'actor_only_transfer':True,'critic_optimizer_normalizer_temperature_and_replay_fresh':True,
 'physics_integration_hz':400,'control_fps':20,'training_compute_dtype':'float32','numeric_guard_training_only':True,
 'previous_inputs_and_videos_unchanged':True,'training_started':False,'policy_improvement_verified':False})
print('Stable-contact cup continuation prepared and CPU actor-transfer contract passed:',R,flush=True)
