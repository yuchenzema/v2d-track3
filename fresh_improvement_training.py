"""Independent fresh FlashSAC experiment with zero initial residual mean.

This tests whether transfer from the pot/lid actor hinders different initially
held objects. Baseline geometry, video timing and evaluation criteria are fixed.
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
from omegaconf import OmegaConf
import flash_chord.training.flash_sac.runner as runner

ROOT = Path('/episode')
episode = json.loads((ROOT/'episode-config.json').read_text())
if episode.get('training_numeric_failure_guard'):
    from training_numeric_failure_guard import install as install_numeric_guard
    install_numeric_guard()
BUDGET = int(episode.get('training_budget',50_000_896))
reference = f"/episode/human_motion_data/{episode['training_reference']}/processed/sequence_id={episode['sequence']}/robot_name=sharpa_wave"


class Metrics:
    def __init__(self):
        self.stream = (ROOT/'flash_chord_coherent_actor_transfer/metrics.jsonl').open('a', buffering=1)
    def log(self, metrics, step):
        if int(step) > 500_000:
            for key in ['sac/actor_loss_mean_per_update', 'sac/critic_loss_mean_per_update', 'policy/temperature_mean_per_update']:
                if key in metrics and not math.isfinite(float(metrics[key])):
                    raise RuntimeError(f'Non-finite fresh experiment learning signal: {key}')
        self.stream.write(json.dumps({'environment_steps': int(step), 'additional_environment_steps': int(step),
                                     'utc': time.strftime('%Y-%m-%dT%H:%M:%SZ', time.gmtime()), **dict(metrics)})+'\n')
    def close(self):
        self.stream.close()


original_run = runner.run_training
def run(*args, **kwargs):
    if episode.get('training_curriculum')=='fresh_rebuild':
        import numpy as np
        environment=args[0] if args else kwargs['env'];e=environment.env
        assert e.config.voc.max_force==0. and e.config.voc.max_torque==0.
        assert e.reset_policy.config.voc_decay_steps==0 and np.all(e.object_control_scale.numpy()==0.)
        exact=int(np.sum(e.reset_policy.finger_scale.numpy()==1.))
        assert e.world_count//4<=exact<=3*e.world_count//4
        (ROOT/'research/fresh-rebuild-training-runtime.json').write_text(json.dumps({
            'world_count':e.world_count,'zero_object_assistance_every_world':True,
            'object_force_limit_N':0.,'object_torque_limit_Nm':0.,
            'exact_reference_finger_worlds':exact,'randomized_finger_worlds':e.world_count-exact},indent=2))
    sink = Metrics()
    kwargs['logger'] = sink
    try:
        return original_run(*args, **kwargs)
    finally:
        sink.close()
runner.run_training = run
sys.argv = [sys.argv[0], 'experiment=sharpa_flash_sac', f"task.parquet='{reference}'",
            'task.motion_speed=0.5', f"sim.substeps={episode.get('physics_substeps',5)}",
            'sim.njmax=1536', 'reset.seed=43', 'scene.world_count=1024', 'logging.mode=disabled',
            'training.replay.capacity=262144', 'training.replay.minimum_size=25600',
            'training.updates_per_collection=4.0', 'training.compute_dtype=bfloat16',
            'training.resume=false', 'training.checkpoint_path=null',
            'training.network.actor_head_mode=residual', 'training.network.residual_mean_head=zero_dense',
            'training.network.initial_normalized_std=0.1', 'training.network.log_std_max=-1.38629436112',
            'training.optimization.residual_mean_learning_rate_multiplier=1.0',
            'training.exploration.target_sigma=0.15', 'reset.reset_finger_openness=0.0',
            f'training.total_environment_steps={BUDGET}',
            f"training.checkpoint.save_interval_environment_steps={min(10000000,max(512,BUDGET//2)) if episode.get('experimental_training') else 10000000}",
            'output_dir=/episode/flash_chord_coherent_actor_transfer',
            'hydra.run.dir=/episode/hydra/fresh-improvement-training']
saved = sorted((ROOT/'flash_chord_coherent_actor_transfer').glob('state_*.safetensors'),
               key=lambda p:int(p.stem.split('_')[-1]))
saved = [p for p in saved if p.with_name(p.name.replace('state_','policy_')).exists()]
if saved:
    own = '/episode/flash_chord_coherent_actor_transfer/' + saved[-1].name
    sys.argv = [arg.replace('training.resume=false','training.resume=true').replace('training.checkpoint_path=null',f'training.checkpoint_path={own}') for arg in sys.argv]
    sys.argv += ['training.checkpoint.load_optimizer=true','training.checkpoint.load_reward_normalizer=true']
    print('Resuming the same independent experiment from',own,flush=True)
if episode.get('training_curriculum')=='fresh_rebuild':
    sys.argv=[arg.replace('reset.reset_finger_openness=0.0','reset.reset_finger_openness=1.0') for arg in sys.argv]
    sys.argv+=['reset.voc_decay_steps=0','env.voc_scale=0.0','env.voc.max_force=0.0','env.voc.max_torque=0.0']
if episode.get('validated_video_pipeline'):
    from legacy_source_frame_playback import install as install_source_frame_playback
    install_source_frame_playback()
native = runpy.run_path('/workspace/scripts/train_flash_sac.py', run_name='native_training')
if episode.get('training_curriculum')=='fresh_rebuild':
    from training_finger_reset import install as install_training_fingers
    install_training_fingers(native['main'].__wrapped__.__globals__)


@hydra.main(version_base='1.3', config_path='/workspace/src/flash_chord/configs', config_name='train_flash_sac')
def main(cfg):
    validated=bool(episode.get('validated_video_pipeline'))
    if validated:
        assert episode.get('training_curriculum')=='fresh_rebuild'
        assert not episode.get('training_focus'),'Validated recipe requires native 50% frame-zero resets, without an overriding focus sampler'
        cfg.action.input_mode='normalized'
        cfg.action.normalized_mapping='linear'
        cfg.action.wrist_pos_scale=.015
        cfg.action.wrist_pos_clip=.015
        cfg.action.finger_joint_pos_scale=.08
        cfg.action.finger_joint_pos_clip=.08
        cfg.reset.always_reset_to_first_frame=False
        cfg.reset.reset_to_first_frame_probability=.5
        cfg.task.motion_speed=1.
        cfg.task.source_frame_playback=True
        assert cfg.env.voc.max_force==0. and cfg.env.voc.max_torque==0. and cfg.env.voc_scale==0.
    first = OmegaConf.to_container(cfg.curriculum.stages[0], resolve=True)
    stages = []
    schedule=([(0.,.35),(0.,.5),(0.,.65),(0.,.8),(0.,.9),(0.,.95)] if episode.get('training_curriculum')=='fresh_rebuild' else
              [(.5, 0.), (.1, .1), (.025, .2), (0., .35), (0., .6), (0., .7)])
    if validated:schedule=[(0.,.5)]*6
    for voc, first_probability in schedule:
        stage = json.loads(json.dumps(first))
        stage['objective_weights']['hand_keypoints'] = 1.
        stage['objective_weights']['hand_joint_pos'] = .5
        stage['voc_scale'], stage['reset_to_first_frame_probability'] = voc, first_probability
        stages.append(stage)
    cfg.curriculum.thresholds = [5_000_000,10_000_000,15_000_000,25_000_000,40_000_000,BUDGET]
    if validated:cfg.curriculum.thresholds=[round(BUDGET*x) for x in [.1,.2,.3,.5,.8,1.]]
    cfg.curriculum.stages = stages
    provenance = {'episode': episode['sequence'], 'engine': 'Newton/FlashCHORD', 'algorithm': 'FlashSAC',
                  'initialization': 'Independent fresh actor, critic, target critic, temperature and optimizer; zero initial residual mean',
                  'initial_mean_head': 'zero_dense', 'preferred_prior_policy': None, 'prior_counter': 0,
                  'additional_training_step_budget': BUDGET, 'world_count': 1024,
                  'baseline_run': episode['baseline_run'], 'baseline_inputs_copied_without_pose_or_geometry_edits': True,
                  'hypothesis': 'Test negative transfer from the earlier pot/lid actor on different grasped objects',
                  'seed': 43, 'motion_speed': .5, 'evaluation_ground_truth_used': False}
    if episode.get('training_curriculum')=='fresh_rebuild':
        provenance.update(training_object_force_limit_N=0.,training_object_torque_limit_Nm=0.,
                          finger_initialization='half_exact_reference_half_uniform_scale',
                          source_clock_preserved=True,reconstruction_rerun_from_original_video=True,
                          hypothesis='Test rebuilt source geometry, corrected anatomical hand tracks, feasible support/contact targets and an independently initialized policy')
    if validated:
        provenance.update(validated_video_pipeline=True,frame_zero_reset_probability=.5,
                          wrist_residual_limit_m=.015,finger_residual_limit_rad=.08,
                          public_validation_manifest=episode.get('public_validation_manifest'),
                          time_advance_frames=episode.get('time_advance_frames',0.),motion_speed=1.,source_frame_playback=True)
    (ROOT/'coherent-training-config.json').write_text(json.dumps(OmegaConf.to_container(cfg,resolve=True),indent=2))
    (ROOT/'coherent-training-provenance.json').write_text(json.dumps(provenance,indent=2))
    files = [ROOT/'reconstruction/scene_motion_physical.npz', Path(reference)/'motion.parquet',
             *sorted((ROOT/'reconstruction/mesh_priors').glob('*.urdf')),
             *sorted((ROOT/'reconstruction/mesh_priors').glob('*.ply')),
             *sorted((ROOT/'reconstruction/mesh_priors').glob('*.obj')),
             *sorted((Path(reference).parents[2]/'reconstructed_stage').glob('*.usda'))]
    previous = ROOT/'coherent-input-checksums.json'
    if previous.exists():
        for name, expected in json.loads(previous.read_text()).items():
            assert hashlib.sha256((ROOT/name).read_bytes()).hexdigest()==expected, ('Frozen experiment input changed',name)
    (ROOT/'coherent-input-checksums.json').write_text(json.dumps({str(p.relative_to(ROOT)):hashlib.sha256(p.read_bytes()).hexdigest() for p in files},indent=2))
    if os.environ.get('V2D_VALIDATE_CONFIG_ONLY') == '1':
        from flash_chord.training.flash_sac.config import NetworkConfig
        NetworkConfig(**{k:v for k,v in OmegaConf.to_container(cfg.training.network,resolve=True).items() if k!='_target_'})
        print('Fresh experiment configuration validated without starting training.',flush=True)
        return
    native['main'].__wrapped__(cfg)


main()
if os.environ.get('V2D_VALIDATE_CONFIG_ONLY') != '1':
    (ROOT/'flash_chord_coherent_actor_transfer/TRAINING_COMPLETED').write_text('Independent fresh improvement experiment completed.\n')
