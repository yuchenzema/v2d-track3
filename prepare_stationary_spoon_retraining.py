"""Prepare an isolated continuation using the video-corrected stationary spoon hold."""
import hashlib,json,os,shutil,stat
from pathlib import Path
from improvement_common import BASE,save,utc
from batch_native import _docker
old=BASE/'runs/episode_000016_newton_grasp_curriculum_20261020';R=BASE/'runs/episode_000016_newton_stationary_hold_20261020';ref='track3_stationary_spoon_hold_probe';STEP=365_000_192
assert not R.exists(),'Keep a prepared continuation immutable; inspect before retrying'
manifest=json.loads((old/'coherent-input-checksums.json').read_text())
for p,h in manifest.items():assert hashlib.sha256((old/p).read_bytes()).hexdigest()==h,p
audit=json.loads((old/f'evaluation/{ref}-collision-audit.json').read_text());assert audit['source_frames_over_10mm']==0
rest=json.loads((old/'evaluation/stationary-spoon-held103-rest.json').read_text());assert all(x['passed_1cm_0_2rad_rest_test'] for x in rest['objects'])
R.mkdir()
for n in ['logs','research','evaluation','videos','deliverables','flash_chord_coherent_actor_transfer']:(R/n).mkdir()
for n in ['reconstruction','scripts']:shutil.copytree(old/n,R/n)
for n in [ref,'track3_video']:shutil.copytree(old/'human_motion_data'/n,R/'human_motion_data'/n)
(R/'.venv-video').symlink_to(BASE/'runs/episode_000005_newton_20261015/.venv-video',target_is_directory=True)
cfg=json.loads((old/'episode-config.json').read_text());cfg.update(run=str(R),native_container='v2d-stationary-episode16-newton',
 training_reference=ref,training_initial_counter=STEP,training_budget=35_008_000,contact_training_additional_budget=35_008_000,
 training_seed_checkpoint_run=str(old),training_variant='stationary_hold_20261020',
 improvement_experiment='Video-corrected stationary spoon hold, partial finger opening and full-state policy continuation')
save(R/'episode-config.json',cfg)
initial=R/'research/initial-policy';initial.mkdir();hashes={}
for kind in ['policy','state']:
 source=old/f'flash_chord_coherent_actor_transfer/{kind}_{STEP}.safetensors';assert source.exists()
 for target in [initial/source.name,R/'flash_chord_coherent_actor_transfer'/source.name]:
  shutil.copy2(source,target);assert hashlib.sha256(source.read_bytes()).hexdigest()==hashlib.sha256(target.read_bytes()).hexdigest()
 hashes[source.name]=hashlib.sha256(source.read_bytes()).hexdigest()
for name in ['source-human-video-recheck-20261020.json','stationary-spoon-hold-probe.json','source-video-review.json']:
 if (old/'research'/name).exists():shutil.copy2(old/'research'/name,R/'research'/name)
for name in ['reference-source-fidelity.json','physical-reference-source-fidelity.json','stationary-spoon-reference-source-fidelity.json','stationary-spoon-held103-rest.json',f'{ref}-collision-audit.json']:
 shutil.copy2(old/'evaluation'/name,R/'evaluation'/name)
selection=json.loads((old/'research/physical-grasp-reference-selection.json').read_text());selection.update(reference=ref,updated_utc=utc(),
 training_initial_counter=STEP,verified_policy_improvement=False,approved_for_independent_training=True,
 maximum_cross_body_penetration_mm=audit['maximum_cross_body_penetration_mm'],
 native_collision_audit=f'evaluation/{ref}-collision-audit.json',terminal_support_evidence='evaluation/stationary-spoon-held103-rest.json',
 physical_reference_source_report='evaluation/stationary-spoon-reference-source-fidelity.json',
 remaining_problem='The previous actor still fails this corrected reference; independent continuation and whole-video validation are required.')
save(R/'research/physical-grasp-reference-selection.json',selection)
save(R/'research/contact-finetune-adjustments.json',{'updated_utc':utc(),'source_interval':[78,142],'residual_mean_learning_rate_multiplier':.1,
 'reason':'Learn the video-observed insertion, stationary supported hold, three-finger opening and final thumb-index release.',
 'new_reference_isolated_from_previous_training':True,'object_assistance':0.,'policy_improvement_verified':False})
for source,target in [('contact_finetune_training.py','train_coherent_actor_transfer.py'),('training_finger_reset.py','training_finger_reset.py'),('validation_stop.py','validation_stop.py')]:shutil.copy2(BASE/source,R/'scripts'/target)
info=json.loads(_docker(['inspect','v2d-grasp-episode16-newton']).stdout)[0]
args=['run','-d','--gpus','all','--runtime','nvidia','--ipc','host','--name',cfg['native_container']]
for m in info['Mounts']:
 source=str(R) if m['Destination']=='/episode' else m['Source'];args+=['-v',source+':'+m['Destination']+('' if m['RW'] else ':ro')]
for node in sorted([*Path('/dev').glob('nvidia*'),*Path('/dev/nvidia-caps').glob('*')]):
 if stat.S_ISCHR(node.stat().st_mode):args+=['--device',str(node)]
args+=['-e','PYGLET_HEADLESS=true','-e','XLA_PYTHON_CLIENT_PREALLOCATE=false','--workdir','/workspace','--entrypoint','/bin/sleep',info['Image'],'infinity']
_docker(args,R/'logs/create-newton.log')
_docker(['exec',cfg['native_container'],'env','CUDA_VISIBLE_DEVICES=','JAX_PLATFORMS=cpu','V2D_VALIDATE_CONFIG_ONLY=1',
 'python','-u','/episode/scripts/train_coherent_actor_transfer.py'],R/'logs/continuation-config-validation.log')
for p,h in manifest.items():assert hashlib.sha256((old/p).read_bytes()).hexdigest()==h,p
save(R/'research/preparation-status.json',{'updated_utc':utc(),'status':'prepared_and_configuration_validated',
 'old_run':str(old),'checkpoint_environment_steps':STEP,'seed_checkpoint_sha256':hashes,'remaining_training_steps':35_008_000,
 'full_learner_state_seeded':True,'previous_training_inputs_unchanged':True,'training_started':False,'policy_improvement_verified':False})
print('Stationary-hold continuation prepared and configuration validated:',R,flush=True)
