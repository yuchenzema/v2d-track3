"""Finetune an existing episode actor on reviewed contact geometry and source motion."""
import hashlib,json,math,os,runpy,sys,time
from pathlib import Path
import hydra,numpy as np
from omegaconf import OmegaConf,open_dict
import flash_chord.training.flash_sac.runner as runner
from training_finger_reset import install
from validation_stop import TrainingValidated,maybe_validation_stop
VALIDATED_STOP_PROTOCOL=2
R=Path('/episode');episode=json.loads((R/'episode-config.json').read_text())
assert episode['episode'] in [10,16,30,32] and episode['training_curriculum']=='contact_finetune'
START=int(episode['training_initial_counter']);ADDITIONAL=int(episode.get('contact_training_additional_budget',50_000_896));END=START+ADDITIONAL
WORLDS=int(episode.get('training_world_count',512));assert WORLDS in [512,1024]
assert ADDITIONAL>=5_000_192 and ADDITIONAL%512==0
folder=R/'flash_chord_coherent_actor_transfer';folder.mkdir(exist_ok=True)
reference=R/f"human_motion_data/{episode['training_reference']}/processed/sequence_id={episode['sequence']}/robot_name=sharpa_wave"
saved=sorted(folder.glob('state_*.safetensors'),key=lambda p:int(p.stem.split('_')[-1]))
saved=[p for p in saved if p.with_name(p.name.replace('state_','policy_')).exists()]
checkpoint=saved[-1] if saved else R/f'research/initial-policy/state_{START}.safetensors'
assert checkpoint.exists();same=bool(saved) or bool(episode.get('contact_full_state_seed',False))
from flash_chord.training.checkpoint import read_checkpoint_metadata
initial_metadata=read_checkpoint_metadata(R/f'research/initial-policy/policy_{START}.safetensors')
initial_network=json.loads(initial_metadata['resolved_config_json'])['training']['network']
assert initial_network['actor_head_mode']=='residual'
adjustment_path=R/'research/contact-finetune-adjustments.json'
adjustments=json.loads(adjustment_path.read_text()) if adjustment_path.exists() else {}
precision=adjustments.get('training_translation_precision')
if episode.get('freeze_actor_normalization'):
 from freeze_actor_normalization import install as install_normalization_freeze
 install_normalization_freeze()
if precision:
 from translation_precision_reward import install as install_precision
 install_precision(**precision)
if episode.get('training_numeric_failure_guard'):
 from training_numeric_failure_guard import install as install_numeric_guard
 install_numeric_guard()
mean_rate=float(adjustments.get('residual_mean_learning_rate_multiplier',1.))
assert 0.<mean_rate<=1.
original_restore=runner.restore_learner_checkpoint
def restore(path,learner,**kwargs):
 if not kwargs.get('load_reward_normalizer',True):
  # Deployable actor checkpoints have no world-shaped exploration arrays.
  # Load only those weights when moving from 1024 to 512 training worlds.
  import jax.numpy as jnp
  from flash_chord.training.flash_sac.checkpoint import load_actor_checkpoint,actor_inference_state
  policy=Path(path).with_name(Path(path).name.replace('state_','policy_',1))
  actor,metadata=load_actor_checkpoint(policy,actor_inference_state(learner))
  assert int(metadata['environment_steps'])==START
  params=actor.params
  if episode.get('reset_residual_head_for_new_reference'):
   assert initial_network['residual_mean_head']=='zero_dense'
   replacements={key:learner.actor.params[key] for key in ['mean','mean_bias','log_std','log_std_bias']}
   from flax.core import FrozenDict
   params=params.copy(replacements) if isinstance(params,FrozenDict) else {**params,**replacements}
  value=learner.replace(actor=learner.actor.replace(params=params,batch_stats=actor.batch_stats),
   environment_steps=jnp.asarray(START,dtype=learner.environment_steps.dtype))
  if os.environ.get('V2D_VALIDATE_ACTOR_TRANSFER_ONLY')=='1':
   import jax
   assert all(np.isfinite(np.asarray(v)).all() for v in jax.tree.leaves(value.actor.params))
   report={'actor_transfer_contract_passed':True,'policy_checkpoint':str(policy),'environment_steps':START,
    'validation_world_count':8,'new_world_shaped_learner_state_retained':True,
    'training_started':False,'training_inputs_frozen':False,'object_assistance':0.}
   if episode.get('reset_residual_head_for_new_reference'):
    assert all(np.all(np.asarray(v)==0) for v in jax.tree.leaves(value.actor.params['mean']))
    assert np.all(np.asarray(value.actor.params['mean_bias'])==0)
    report['residual_mean_reset_exactly_zero']=True
   (R/'research/contact-actor-transfer-contract.json').write_text(json.dumps(report,indent=2))
   print(json.dumps(report),flush=True);raise SystemExit(0)
  return value,metadata
 return original_restore(path,learner,**kwargs)
runner.restore_learner_checkpoint=restore
class Metrics:
 def __init__(self):self.stream=(folder/'metrics.jsonl').open('a',buffering=1)
 def log(self,metrics,step):
  for k in ['sac/actor_loss_mean_per_update','sac/critic_loss_mean_per_update','policy/temperature_mean_per_update']:
   if step>START+500_000 and k in metrics and not math.isfinite(float(metrics[k])):raise RuntimeError('Nonfinite finetuning signal: '+k)
  if hasattr(self,'reset_policy'):
   probability=float(episode.get('contact_focus_probability',.25 if step<START+15_000_576 else .10))
   self.reset_policy.focus_probability.assign(np.array([probability],np.float32))
   current=json.loads(adjustment_path.read_text()) if adjustment_path.exists() else adjustments
   source_interval=current.get('source_interval',FOCUS['source_interval'])
   assert 0<=source_interval[0]<=source_interval[1]<episode['frames']
   native_interval=[round(v*(self.reference_frames-1)/(episode['frames']-1)) for v in source_interval]
   if native_interval!=[self.reset_policy.focus_lo,self.reset_policy.focus_hi]:self.reset_policy.set_focus_interval(*native_interval)
   metrics={**metrics,'curriculum/contact_focus_reset_probability':probability,
    'curriculum/contact_focus_source_start':source_interval[0],'curriculum/contact_focus_source_end':source_interval[1]}
   env=self.reset_policy.environment
   if hasattr(env,'_numeric_failure_counts'):
    counts=env._numeric_failure_counts.numpy()
    metrics={**metrics,'numerical_guard/failure_resets_total':int(counts.sum()),'numerical_guard/maximum_resets_in_one_world':int(counts.max())}
  self.stream.write(json.dumps({'environment_steps':int(step),'additional_environment_steps':int(step)-START,
   'training_phase':'full','utc':time.strftime('%Y-%m-%dT%H:%M:%SZ',time.gmtime()),**dict(metrics)})+'\n')
  maybe_validation_stop(R,int(step),START)
 def close(self):self.stream.close()
original_run=runner.run_training
def run(*args,**kwargs):
 env=(args[0] if args else kwargs['env']).env
 assert env.reset_policy.config.voc_decay_steps==0 and env.config.voc.max_force==0. and env.config.voc.max_torque==0.
 assert np.all(env.object_control_scale.numpy()==0.)
 exact=int(np.sum(env.reset_policy.finger_scale.numpy()==1.))
 assert env.world_count//4<=exact<=3*env.world_count//4
 for hand in env.scene.layout.hands:
  ids=list(hand.wrist_pos_dof_ids);expected=episode.get('wrist_side_motor_profiles',{}).get(hand.side,episode.get('wrist_motor_profile',{'kp':25.,'kd':6.}))
  for attribute,key in [('joint_target_ke','kp'),('joint_target_kd','kd')]:
   values=getattr(env.model,attribute).numpy().reshape(env.world_count,env.num_joint_dof)
   assert np.allclose(values[:,ids],expected[key])
  limits=env.model.joint_effort_limit.numpy().reshape(env.world_count,env.num_joint_dof)
  assert np.allclose(limits[:,ids],45.)
 (R/'research/contact-finetune-zero-support-runtime.json').write_text(json.dumps({
  'world_count':env.world_count,'reference_frames':env.reference.num_frames,'exact_reference_finger_worlds':exact,
  'reset_settling_steps':0,'object_force_N':0.,'object_torque_Nm':0.,'object_control_scale_zero_every_world':True},indent=2))
 sink=Metrics();sink.reset_policy=env.reset_policy;sink.reference_frames=env.reference.num_frames;kwargs['logger']=sink
 try:return original_run(*args,**kwargs)
 finally:sink.close()
runner.run_training=run
sys.argv=[sys.argv[0],'experiment=sharpa_flash_sac',f"task.parquet='{reference}'",'task.motion_speed=0.5',
 f"sim.substeps={episode.get('physics_substeps',10)}",f"sim.njmax={episode.get('physics_njmax',4096 if episode['episode']==10 else 1536)}",f"sim.nconmax={episode.get('physics_nconmax',2048 if episode['episode']==10 else 512)}",f'scene.world_count={WORLDS}',f"reset.seed={episode.get('training_seed',46)}",'logging.mode=disabled',
 *(['scene.decompose_objects=false'] if episode.get('predecomposed_object_collisions') else []),
 f"training.replay.capacity={episode.get('contact_replay_capacity',262144)}",f"training.replay.minimum_size={episode.get('contact_replay_minimum_size',25600)}",'training.updates_per_collection=4.0',
 f"training.compute_dtype={episode.get('training_compute_dtype','bfloat16')}",'training.network.actor_head_mode=residual',f"training.network.residual_mean_head={initial_network['residual_mean_head']}",
 'training.network.initial_normalized_std=0.05','training.network.log_std_max=-1.38629436112',
 f'training.optimization.residual_mean_learning_rate_multiplier={mean_rate}',f"training.optimization.actor_learning_rate_multiplier={episode.get('training_actor_learning_rate_multiplier',1.)}",f"training.optimization.behavior_cloning_coefficient={episode.get('training_behavior_cloning_coefficient',0.)}",'training.exploration.target_sigma=0.1',
 'reset.reset_finger_openness=1.0','reset.voc_decay_steps=0','env.voc_scale=0.0','env.voc.max_force=0.0','env.voc.max_torque=0.0',
 'training.resume=true',f'training.checkpoint_path={checkpoint}',f'training.checkpoint.load_optimizer={str(same).lower()}',
 f'training.checkpoint.load_reward_normalizer={str(same).lower()}',f'training.total_environment_steps={END}',
 f"training.checkpoint.save_interval_environment_steps={episode.get('contact_checkpoint_interval',5_000_000)}",f'output_dir={folder}','hydra.run.dir=/episode/hydra/contact-finetuning']
native=runpy.run_path('/workspace/scripts/train_flash_sac.py',run_name='native_training')
FOCUS={'source_interval':adjustments.get('source_interval',[52,115] if episode['episode']==10 else [84,134]),'source_frame_count':episode['frames'],'probability':float(episode.get('contact_focus_probability',.25))}
install(native['main'].__wrapped__.__globals__,focus=FOCUS)
@hydra.main(version_base='1.3',config_path='/workspace/src/flash_chord/configs',config_name='train_flash_sac')
def main(cfg):
 validation_transfer=os.environ.get('V2D_VALIDATE_ACTOR_TRANSFER_ONLY')=='1'
 if validation_transfer:
  cfg.scene.world_count=8;cfg.training.replay.capacity=512;cfg.training.replay.minimum_size=256;cfg.training.replay.batch_size=256
 from flash_chord.embodiments.sharpa_hands import SharpaHandsConfig
 overrides=dict(SharpaHandsConfig().joint_overrides)
 if episode.get('wrist_motor_profile'):
  for side in ['left','right']:overrides[side+r'_wrist_[xyz]$']=episode['wrist_motor_profile']
 for side,profile in episode.get('wrist_side_motor_profiles',{}).items():
  assert side in ['left','right'] and set(profile)=={'kp','kd'} and 25<=profile['kp']<=500 and 6<=profile['kd']<=40
  overrides[side+r'_wrist_[xyz]$']=profile
 with open_dict(cfg.embodiment.config):cfg.embodiment.config.joint_overrides=overrides
 with open_dict(cfg):cfg.training_reset_initialization={'finger_pose_policy':'half_exact_reference_half_uniform_scale',
  'exact_reference_probability':.5,'inference':'native explicit exact-reference reset, no reset hold or object assistance'}
 first=OmegaConf.to_container(cfg.curriculum.stages[0],resolve=True);stages=[]
 for probability in [.65,.8,.9]:
  probability=float(episode.get('contact_reset_first_frame_probability',probability))
  value=json.loads(json.dumps(first));value.update(voc_scale=0.,reset_to_first_frame_probability=probability)
  value['objective_weights'].update(hand_keypoints=1.,hand_joint_pos=.5);stages.append(value)
 cfg.curriculum.stages=stages
 first_stage=min(5_000_192,(ADDITIONAL//4//512)*512)
 second_stage=min(20_000_768,(3*ADDITIONAL//4//512)*512)
 cfg.curriculum.thresholds=[START+first_stage,START+second_stage,END]
 configuration=OmegaConf.to_container(cfg,resolve=True)
 if not validation_transfer:(R/'coherent-training-config.json').write_text(json.dumps(configuration,indent=2))
 provenance={'episode':episode['sequence'],'engine':'Newton/FlashCHORD','algorithm':'FlashSAC',
  'initialization':'Transfer this episode actor; reset critic, temperature, normalizer, exploration and optimizer at the new contact model. Resume own complete states thereafter.',
  'prior_counter':START,'additional_training_step_budget':ADDITIONAL,'training_phase':'full','world_count':WORLDS,'seed':46,
  'baseline_run':episode['baseline_run'],'reference':episode['training_reference'],'motion_speed':.5,
  'reset_settling_steps':0,'training_object_force_limit_N':0.,'training_object_torque_limit_Nm':0.,
  'historical_warmstart_may_include_supported_training':True,'evaluation_ground_truth_used':False,
  'evaluation_object_assistance':0.,'wrist_side_motor_profiles':episode.get('wrist_side_motor_profiles',{}),
  'reference_and_geometry_review':'research/physical-grasp-reference-selection.json','baseline_artifacts_unchanged':True}
 provenance.update(residual_mean_learning_rate_multiplier=mean_rate,training_adjustments_file=str(adjustment_path.relative_to(R)))
 provenance.update(actor_learning_rate_multiplier=float(episode.get('training_actor_learning_rate_multiplier',1.)),seed=episode.get('training_seed',46))
 provenance.update(freeze_actor_normalization=bool(episode.get('freeze_actor_normalization')),behavior_cloning_coefficient=float(episode.get('training_behavior_cloning_coefficient',0.)))
 provenance['reset_residual_head_for_new_reference']=bool(episode.get('reset_residual_head_for_new_reference'))
 if episode.get('reset_residual_head_for_new_reference'):
  provenance['initialization']='Transfer pretrained actor features and normalization; reset the residual mean and exploration heads, critics, optimizer, normalizer and replay for the corrected reference. Resume complete new learner states thereafter.'
 provenance['actual_resume_checkpoint']={'path':str(checkpoint.relative_to(R)),
  'sha256':hashlib.sha256(checkpoint.read_bytes()).hexdigest(),'optimizer_critic_temperature_and_global_normalizer_restored':same,
  'replay_restarted':True}
 if episode.get('training_seed_checkpoint_run'):
  provenance['initialization']='Resume the preserved full learner checkpoint from the previous reference: actor, critics, optimizer, temperature, exploration and normalizer. Replay is rebuilt from the corrected reference.'
  provenance['training_seed_checkpoint_run']=episode['training_seed_checkpoint_run']
 provenance['training_only_translation_precision']=precision
 provenance['training_numeric_failure_guard']=bool(episode.get('training_numeric_failure_guard'))
 provenance['training_compute_dtype']=episode.get('training_compute_dtype','bfloat16')
 if precision:
  provenance['training_objective_extension_sha256']=hashlib.sha256((R/'scripts/translation_precision_reward.py').read_bytes()).hexdigest()
  provenance['evaluation_objective']='Unmodified native FlashCHORD objective; extension imported only by this training entry point.'
 provenance['training_only_contact_focus_resets']={**FOCUS,'probability_after_15M_new_steps':.10,
  'inference':'Independent whole-episode evaluation starts at source frame0 with exact reference fingers and zero assistance.'}
 if not validation_transfer:(R/'coherent-training-provenance.json').write_text(json.dumps(provenance,indent=2))
 files=[R/'reconstruction/scene_motion_physical.npz',reference/'motion.parquet',
  *sorted((R/'reconstruction/mesh_priors').glob('*.urdf')),*sorted((R/'reconstruction/mesh_priors').glob('*.ply')),
  *sorted((R/'reconstruction/mesh_priors').glob('*.obj')),*sorted((reference.parents[2]/'reconstructed_stage').glob('*.usda'))]
 checksums={str(p.relative_to(R)):hashlib.sha256(p.read_bytes()).hexdigest() for p in files};frozen=R/'coherent-input-checksums.json'
 if frozen.exists():assert json.loads(frozen.read_text())==checksums,'Frozen contact inputs changed'
 if os.environ.get('V2D_VALIDATE_CONFIG_ONLY')=='1':
  from flash_chord.training.flash_sac.config import NetworkConfig
  NetworkConfig(**{k:v for k,v in configuration['training']['network'].items() if k!='_target_'})
  assert configuration['env']['reset']['voc_decay_steps']==0 and configuration['env']['voc']['max_force']==0.
  print('Validated strict contact finetuning configuration without starting training.',flush=True);return
 if not validation_transfer:frozen.write_text(json.dumps(checksums,indent=2))
 try:native['main'].__wrapped__(cfg)
 except TrainingValidated as decision:
  record=json.loads((R/'research/training-early-stop.json').read_text())
  provenance.update(planned_additional_training_step_budget=ADDITIONAL,
   additional_training_step_budget=record['actual_additional_environment_steps'],validated_early_stop=True,
   training_stop_decision='research/training-early-stop.json')
  (R/'coherent-training-provenance.json').write_text(json.dumps(provenance,indent=2));print(decision,flush=True)
main()
if os.environ.get('V2D_VALIDATE_CONFIG_ONLY')!='1':
 (folder/'TRAINING_COMPLETED').write_text('Contact finetuning completed; whole-source rollout validation is separate.\n')
