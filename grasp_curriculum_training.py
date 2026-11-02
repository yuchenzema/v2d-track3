"""Learn the source opening grasps, then train the full unchanged episode.

Only training windows/curriculum differ. Published evaluations always use the
full original reference, original geometry, zero object assistance and no resets.
"""
import hashlib
import json
import math
import os
from pathlib import Path
import runpy
import sys
import time
import hydra
from omegaconf import OmegaConf,open_dict
import flash_chord.training.flash_sac.runner as runner

ROOT=Path('/episode')
episode=json.loads((ROOT/'episode-config.json').read_text())
selection=ROOT/'research/physical-grasp-reference-selection.json'
if selection.exists():
    selected=json.loads(selection.read_text())
    if selected.get('approved_for_independent_training'):
        episode.update(training_reference=selected['reference'],wrist_motor_profile=selected['wrist_motor_profile'],
            physical_grasp_refinement=str(selection.relative_to(ROOT)))
        (ROOT/'episode-config.json').write_text(json.dumps(episode,indent=2))
PREFIX_BUDGET=15_000_576
TOTAL_BUDGET=65_001_472
validate=os.environ.get('V2D_VALIDATE_CONFIG_ONLY')=='1'
prefix_finished=(ROOT/'grasp_pretraining/TRAINING_COMPLETED').exists()
stage=os.environ.get('V2D_GRASP_STAGE','full' if prefix_finished else 'prefix')
if stage not in ['prefix','full']:raise ValueError('Unknown grasp-curriculum stage')
folder=ROOT/('grasp_pretraining' if stage=='prefix' else 'flash_chord_coherent_actor_transfer')
folder.mkdir(exist_ok=True)
prefix_end=min(episode['frames']-1,max(g['start'] for g in episode['grasps'])+22)
# Native task windows slice the already-resampled, half-speed control grid.
# Include the last intended source sample in that half-open window.
control_frames=2*(episode['frames']-1)
prefix_control_end=int(round(prefix_end*(control_frames-1)/(episode['frames']-1)))+1
reference=ROOT/f"human_motion_data/{episode['training_reference']}/processed/sequence_id={episode['sequence']}/robot_name=sharpa_wave"


class Metrics:
    def __init__(self):self.stream=(ROOT/'flash_chord_coherent_actor_transfer/metrics.jsonl').open('a',buffering=1)
    def log(self,metrics,step):
        if int(step)>500_000 and any(k in metrics and not math.isfinite(float(metrics[k])) for k in
            ['sac/actor_loss_mean_per_update','sac/critic_loss_mean_per_update','policy/temperature_mean_per_update']):
            raise RuntimeError('Non-finite grasp-curriculum learning signal')
        self.stream.write(json.dumps({'environment_steps':int(step),'additional_environment_steps':int(step),
            'training_phase':stage,'utc':time.strftime('%Y-%m-%dT%H:%M:%SZ',time.gmtime()),**dict(metrics)})+'\n')
    def close(self):self.stream.close()


native_run=runner.run_training
def logged_run(*args,**kwargs):
    sink=Metrics();kwargs['logger']=sink
    try:return native_run(*args,**kwargs)
    finally:sink.close()
runner.run_training=logged_run
native_restore=runner.restore_learner_checkpoint
def restore(path,learner,**kwargs):
    restored,metadata=native_restore(path,learner,**kwargs)
    if not kwargs.get('load_reward_normalizer',True):
        restored=restored.replace(critic=learner.critic,target_critic=learner.target_critic,
            temperature=learner.temperature,reward_normalizer=learner.reward_normalizer,
            exploration=learner.exploration,loss_scale=learner.loss_scale)
        print('Opening-grasp actor transferred to full motion; critic/temperature/normalizer/exploration initialized fresh.',flush=True)
    return restored,metadata
runner.restore_learner_checkpoint=restore
saved=sorted(folder.glob('state_*.safetensors'),key=lambda p:int(p.stem.split('_')[-1]))
saved=[p for p in saved if p.with_name(p.name.replace('state_','policy_')).exists()]
checkpoint=saved[-1] if saved else ROOT/f'grasp_pretraining/state_{PREFIX_BUDGET}.safetensors' if stage=='full' else None
same_stage=bool(saved)
budget=PREFIX_BUDGET if stage=='prefix' else TOTAL_BUDGET
sys.argv=[sys.argv[0],'experiment=sharpa_flash_sac',f"task.parquet='{reference}'",'task.motion_speed=0.5',
    f'task.motion_end_frame={prefix_control_end if stage=="prefix" else -1}',f"sim.substeps={episode.get('physics_substeps',5)}",
    'sim.njmax=1536','reset.seed=44','scene.world_count=1024','logging.mode=disabled',
    'training.replay.capacity=262144','training.replay.minimum_size=25600','training.updates_per_collection=4.0',
    'training.compute_dtype=bfloat16','training.network.actor_head_mode=residual','training.network.residual_mean_head=zero_dense',
    'training.network.initial_normalized_std=0.1','training.network.log_std_max=-1.38629436112',
    'training.optimization.residual_mean_learning_rate_multiplier=1.0','training.exploration.target_sigma=0.15',
    'reset.reset_finger_openness=0.0',f'training.total_environment_steps={budget}',
    f'training.checkpoint.save_interval_environment_steps={5_000_000 if stage=="prefix" else 10_000_000}',
    f'training.resume={str(checkpoint is not None).lower()}',f'training.checkpoint_path={checkpoint or "null"}',
    f'training.checkpoint.load_optimizer={str(same_stage).lower()}',
    f'training.checkpoint.load_reward_normalizer={str(same_stage).lower()}',
    f'output_dir={folder}',f'hydra.run.dir=/episode/hydra/grasp-curriculum-{stage}']
native=runpy.run_path('/workspace/scripts/train_flash_sac.py',run_name='native_training')


@hydra.main(version_base='1.3',config_path='/workspace/src/flash_chord/configs',config_name='train_flash_sac')
def main(cfg):
    if episode.get('wrist_motor_profile'):
        from flash_chord.embodiments.sharpa_hands import SharpaHandsConfig
        profile=episode['wrist_motor_profile']
        assert set(profile)=={'kp','kd'} and 25.<=float(profile['kp'])<=500. and 6.<=float(profile['kd'])<=40.
        overrides=dict(SharpaHandsConfig().joint_overrides)
        overrides[r'wrist_[xyz]$']={**overrides[r'wrist_[xyz]$'],**profile}
        with open_dict(cfg.embodiment.config):cfg.embodiment.config.joint_overrides=overrides
    first=OmegaConf.to_container(cfg.curriculum.stages[0],resolve=True)
    schedule=([(.1,.8),(.025,1.),(0.,1.)] if stage=='prefix' else [(0.,.75),(0.,.8),(0.,.9)])
    stages=[]
    for voc,probability in schedule:
        value=json.loads(json.dumps(first));value['voc_scale']=voc;value['reset_to_first_frame_probability']=probability
        value['objective_weights']['hand_keypoints']=1.;value['objective_weights']['hand_joint_pos']=.5
        stages.append(value)
    cfg.curriculum.stages=stages
    cfg.curriculum.thresholds=([3_000_320,5_000_192,PREFIX_BUDGET] if stage=='prefix' else
                              [PREFIX_BUDGET+5_000_192,PREFIX_BUDGET+20_000_768,TOTAL_BUDGET])
    configuration=OmegaConf.to_container(cfg,resolve=True)
    (ROOT/f'research/grasp-{stage}-training-config.json').write_text(json.dumps(configuration,indent=2))
    (ROOT/'coherent-training-config.json').write_text(json.dumps(configuration,indent=2))
    (ROOT/'coherent-training-provenance.json').write_text(json.dumps({
        'episode':episode['sequence'],'engine':'Newton/FlashCHORD','algorithm':'FlashSAC',
        'initialization':'Fresh actor learns the opening grasp; the same learned actor transfers to the unchanged full episode',
        'training_phase':stage,'prior_counter':0,'additional_training_step_budget':TOTAL_BUDGET,
        'prefix_training_steps':PREFIX_BUDGET,'full_motion_additional_steps':TOTAL_BUDGET-PREFIX_BUDGET,
        'prefix_end_source_frame':prefix_end,'prefix_control_end_exclusive':prefix_control_end,'final_evaluation_uses_full_episode':True,
        'world_count':1024,'seed':44,'baseline_run':episode['baseline_run'],
        'baseline_inputs_copied_without_pose_or_geometry_edits':not bool(episode.get('physical_grasp_refinement')),
        'source_object_poses_and_camera_unchanged':True,
        'physical_hand_grasp_reference_refinement':episode.get('physical_grasp_refinement'),
        'wrist_motor_profile':episode.get('wrist_motor_profile'),
        'evaluation_ground_truth_used':False,'evaluation_object_assistance':0.,'motion_speed':.5},indent=2))
    files=[ROOT/'reconstruction/scene_motion_physical.npz',reference/'motion.parquet',
        *sorted((ROOT/'reconstruction/mesh_priors').glob('*.urdf')),
        *sorted((ROOT/'reconstruction/mesh_priors').glob('*.ply')),
        *sorted((ROOT/'reconstruction/mesh_priors').glob('*.obj')),
        *sorted((reference.parents[2]/'reconstructed_stage').glob('*.usda'))]
    checksums={str(p.relative_to(ROOT)):hashlib.sha256(p.read_bytes()).hexdigest() for p in files}
    previous=ROOT/'coherent-input-checksums.json'
    if previous.exists() and json.loads(previous.read_text())!=checksums:raise RuntimeError('Frozen full-reference input changed')
    if validate:
        from flash_chord.training.flash_sac.config import NetworkConfig
        NetworkConfig(**{k:v for k,v in configuration['training']['network'].items() if k!='_target_'})
        assert configuration['task']['motion_end_frame']==(prefix_control_end if stage=='prefix' else -1)
        print('Validated grasp curriculum stage',stage,'without starting policy training.',flush=True);return
    previous.write_text(json.dumps(checksums,indent=2))
    native['main'].__wrapped__(cfg)


main()
if not validate:
    (folder/'TRAINING_COMPLETED').write_text(f'Grasp curriculum {stage} training completed.\n')
    if stage=='prefix':
        os.environ['V2D_GRASP_STAGE']='full'
        os.execv(sys.executable,[sys.executable,'-u',str(ROOT/'scripts/train_coherent_actor_transfer.py')])
