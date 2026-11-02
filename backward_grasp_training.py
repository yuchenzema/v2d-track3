"""Learn holding first, then expand the pickup window backwards to source frame zero."""
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
from validation_stop import TrainingValidated,maybe_validation_stop
VALIDATED_STOP_PROTOCOL=2

ROOT=Path('/episode')
episode=json.loads((ROOT/'episode-config.json').read_text())
selection=ROOT/'research/physical-grasp-reference-selection.json'
if selection.exists():
    selected=json.loads(selection.read_text())
    if selected.get('approved_for_independent_training'):
        episode.update(training_reference=selected['reference'],wrist_motor_profile=selected['wrist_motor_profile'],
            physical_grasp_refinement=str(selection.relative_to(ROOT)))
        if selected.get('backward_hold_source_frame') is not None:
            episode['backward_hold_source_frame']=int(selected['backward_hold_source_frame'])
        if selected.get('wrist_side_motor_profiles'):
            episode['wrist_side_motor_profiles']=selected['wrist_side_motor_profiles']
        if selected.get('strict_reset_retrain_prefix_steps'):
            episode['strict_reset_retrain_prefix_steps']=int(selected['strict_reset_retrain_prefix_steps'])
        if selected.get('adaptive_single_prefix'):
            episode['adaptive_single_prefix']=True
        (ROOT/'episode-config.json').write_text(json.dumps(episode,indent=2))
STAGE_BUDGET=5_000_192
RETRAIN_OFFSET=int(episode.get('strict_reset_retrain_prefix_steps',0))
assert RETRAIN_OFFSET in [0,STAGE_BUDGET]
PREFIX_BUDGET=4*STAGE_BUDGET+RETRAIN_OFFSET
TOTAL_BUDGET=PREFIX_BUDGET+50_000_896
validate=os.environ.get('V2D_VALIDATE_CONFIG_ONLY')=='1'
prefix_finished=(ROOT/'grasp_pretraining/TRAINING_COMPLETED').exists()
stage=os.environ.get('V2D_GRASP_STAGE','full' if prefix_finished else 'prefix')
if stage not in ['prefix','full']:raise ValueError('Unknown grasp-curriculum stage')
prefix_root=ROOT/'grasp_pretraining';prefix_root.mkdir(exist_ok=True)
backward_stage=next((i for i in range(4) if not (prefix_root/f'stage_{i}/TRAINING_COMPLETED').exists()),3)
backward_stage=int(os.environ.get('V2D_BACKWARD_STAGE',backward_stage))
if episode.get('adaptive_single_prefix'):
    assert RETRAIN_OFFSET==0
    backward_stage=3
assert 0<=backward_stage<4
folder=prefix_root/f'stage_{backward_stage}' if stage=='prefix' else ROOT/'flash_chord_coherent_actor_transfer'
folder.mkdir(exist_ok=True)
last_grasp=max(g['start'] for g in episode['grasps'])
hold_start=int(episode.get('backward_hold_source_frame',last_grasp+12))
source_start=[hold_start,last_grasp,max(0,last_grasp-10),0][backward_stage] if stage=='prefix' else 0
prefix_end=min(episode['frames']-1,max(last_grasp+30,hold_start+18))
assert 0<=source_start<prefix_end<episode['frames']
# Native task windows slice the already-resampled, half-speed control grid.
# Include the last intended source sample in that half-open window.
control_frames=2*(episode['frames']-1)
prefix_control_end=int(round(prefix_end*(control_frames-1)/(episode['frames']-1)))+1
start_control=int(round(source_start*(control_frames-1)/(episode['frames']-1)))
reference=ROOT/f"human_motion_data/{episode['training_reference']}/processed/sequence_id={episode['sequence']}/robot_name=sharpa_wave"
ACTIVE_RESET_POLICY=None
adaptive_path=ROOT/'research/adaptive-reset-progress.json'
adaptive_history=json.loads(adaptive_path.read_text()).get('history',[]) if adaptive_path.exists() else []
adaptive=None


class Metrics:
    def __init__(self):self.stream=(ROOT/'flash_chord_coherent_actor_transfer/metrics.jsonl').open('a',buffering=1)
    def log(self,metrics,step):
        if int(step)>500_000 and any(k in metrics and not math.isfinite(float(metrics[k])) for k in
            ['sac/actor_loss_mean_per_update','sac/critic_loss_mean_per_update','policy/temperature_mean_per_update']):
            raise RuntimeError('Non-finite grasp-curriculum learning signal')
        if ACTIVE_RESET_POLICY is not None and getattr(ACTIVE_RESET_POLICY,'adaptive_enabled',False):
            decision=ACTIVE_RESET_POLICY.maybe_advance(int(step))
            if decision is not None:
                adaptive_history.append(decision)
                payload={'episode':episode['episode'],'reference':episode['training_reference'],'history':adaptive_history,
                    'minimum_reference_frame':decision['minimum_reference_frame'],'updated_utc':time.strftime('%Y-%m-%dT%H:%M:%SZ',time.gmtime()),
                    'object_support_force_limit_N':0.,'evaluation_starts_at_original_frame_zero':True}
                temporary=adaptive_path.with_suffix('.tmp');temporary.write_text(json.dumps(payload,indent=2));temporary.replace(adaptive_path)
                if decision['advanced']:print('Adaptive pickup start moved to native frame',decision['minimum_reference_frame'],flush=True)
            minimum=int(ACTIVE_RESET_POLICY.minimum_frame.numpy()[0])
            metrics={**dict(metrics),'curriculum/adaptive_reset_min_reference_frame':minimum,
                'curriculum/adaptive_reset_min_source_frame':minimum*(episode['frames']-1)/(control_frames-1)}
        self.stream.write(json.dumps({'environment_steps':int(step),'additional_environment_steps':int(step),
            'training_phase':stage,'backward_stage':backward_stage if stage=='prefix' else None,
            'training_start_source_frame':source_start,'utc':time.strftime('%Y-%m-%dT%H:%M:%SZ',time.gmtime()),**dict(metrics)})+'\n')
        if stage=='full':maybe_validation_stop(ROOT,int(step),0)
    def close(self):self.stream.close()


native_run=runner.run_training
def logged_run(*args,**kwargs):
    global ACTIVE_RESET_POLICY
    import numpy as np
    environment=args[0] if args else kwargs['env']
    e=environment.env
    ACTIVE_RESET_POLICY=e.reset_policy
    assert e.reset_policy.config.voc_decay_steps==0
    assert e.config.voc.max_force==0. and e.config.voc.max_torque==0.
    assert np.all(e.object_control_scale.numpy()==0.)
    factors=e.reset_policy.finger_scale.numpy()
    exact=int(np.sum(factors==1.))
    assert getattr(e.reset_policy,'exact_reference_finger_probability',None)==.5
    assert e.world_count//4<=exact<=3*e.world_count//4
    (ROOT/f'research/backward-zero-support-{stage}-{backward_stage}.json').write_text(json.dumps({
        'world_count':e.world_count,'reset_settling_steps':0,'object_force_limit_N':0.,'object_torque_limit_Nm':0.,
        'actual_reset_object_control_scale_zero_every_world':True,'reference_frames':e.reference.num_frames,
        'exact_reference_finger_worlds':exact,'scaled_finger_worlds':e.world_count-exact,
        'source_start_frame':source_start,'historical_supported_warmstart_environment_steps':RETRAIN_OFFSET},indent=2))
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
prior_stage=3 if stage=='full' else backward_stage-1
checkpoint=(saved[-1] if saved else prefix_root/f'stage_{prior_stage}/state_{(prior_stage+1)*STAGE_BUDGET+RETRAIN_OFFSET}.safetensors'
            if prior_stage>=0 else None)
if episode.get('adaptive_single_prefix') and stage=='prefix' and not saved:
    # This fresh run learns the entire backward curriculum continuously. No
    # checkpoint or completion marker is invented for skipped fixed phases.
    checkpoint=None
same_stage=bool(saved)
if checkpoint is not None and checkpoint.exists():
    import struct
    with checkpoint.open('rb') as stream:
        size=struct.unpack('<Q',stream.read(8))[0]
        previous_metadata=json.loads(stream.read(size)).get('__metadata__',{})
    previous_cfg=json.loads(previous_metadata.get('user.resolved_config_json','{}'))
    if previous_cfg.get('env',{}).get('reset',{}).get('voc_decay_steps',0)>0:
        same_stage=False
        print('Warmstart used supported reset settling; transfer actor only, reset critic/optimizer, and disable support/delay.',flush=True)
budget=(backward_stage+1)*STAGE_BUDGET+RETRAIN_OFFSET if stage=='prefix' else TOTAL_BUDGET
sys.argv=[sys.argv[0],'experiment=sharpa_flash_sac',f"task.parquet='{reference}'",'task.motion_speed=0.5',
    f'task.motion_start_frame={start_control}',f'task.motion_end_frame={prefix_control_end if stage=="prefix" else -1}',f"sim.substeps={episode.get('physics_substeps',5)}",
    'sim.njmax=1536','reset.seed=44','scene.world_count=1024','logging.mode=disabled',
    'training.replay.capacity=262144','training.replay.minimum_size=25600','training.updates_per_collection=4.0',
    'training.compute_dtype=bfloat16','training.network.actor_head_mode=residual','training.network.residual_mean_head=zero_dense',
    'training.network.initial_normalized_std=0.05','training.network.log_std_max=-1.38629436112',
    'training.optimization.residual_mean_learning_rate_multiplier=1.0','training.exploration.target_sigma=0.1',
    'reset.reset_finger_openness=1.0','reset.voc_decay_steps=0','env.voc_scale=0.0',
    'env.voc.max_force=0.0','env.voc.max_torque=0.0',f'training.total_environment_steps={budget}',
    f'training.checkpoint.save_interval_environment_steps={1_000_000 if stage=="prefix" else 5_000_000}',
    f'training.resume={str(checkpoint is not None).lower()}',f'training.checkpoint_path={checkpoint or "null"}',
    f'training.checkpoint.load_optimizer={str(same_stage).lower()}',
    f'training.checkpoint.load_reward_normalizer={str(same_stage).lower()}',
    f'output_dir={folder}',f'hydra.run.dir=/episode/hydra/backward-grasp-{stage}-{backward_stage}']
if stage=='full' or backward_stage==3:
    # Begin at the independently checked holding pose, then recover pickup one
    # source frame at a time. Starting immediately before contact skips the
    # useful stable-grasp region and can leave all boundary episodes failing.
    minimum=round(hold_start*(control_frames-1)/(episode['frames']-1))
    previous_step=int(checkpoint.stem.split('_')[-1]) if checkpoint is not None else 0
    compatible=[x for x in adaptive_history if x['environment_steps']<=previous_step]
    if compatible:minimum=int(compatible[-1]['minimum_reference_frame'])
    adaptive={'minimum_frame':minimum,'opening_gate_end':prefix_control_end,'stride':2}
native=runpy.run_path('/workspace/scripts/train_flash_sac.py',run_name='native_training')
from training_finger_reset import install as install_training_fingers
install_training_fingers(native['main'].__wrapped__.__globals__,adaptive=adaptive)


@hydra.main(version_base='1.3',config_path='/workspace/src/flash_chord/configs',config_name='train_flash_sac')
def main(cfg):
    with open_dict(cfg):
        cfg.training_reset_initialization={'finger_pose_policy':'half_exact_reference_half_uniform_scale',
            'exact_reference_probability':.5,'uniform_scale_range':[0.,1.],
            'inference':'native explicit exact-reference reset; no inference extension required',
            'adaptive_boundary_reset_overlap':{'boundary_fraction':.7,'two_source_frames_earlier_fraction':.2,
                'original_frame_zero_fraction':.1,'applies_to':'Resets otherwise clamped to the current adaptive boundary; naturally later resets retain their native distribution.'},
            'adaptive_training_reset':adaptive}
    if episode.get('wrist_motor_profile'):
        from flash_chord.embodiments.sharpa_hands import SharpaHandsConfig
        profile=episode['wrist_motor_profile']
        assert set(profile)=={'kp','kd'} and 25.<=float(profile['kp'])<=500. and 6.<=float(profile['kd'])<=40.
        overrides=dict(SharpaHandsConfig().joint_overrides)
        overrides[r'wrist_[xyz]$']={**overrides[r'wrist_[xyz]$'],**profile}
        for side,value in episode.get('wrist_side_motor_profiles',{}).items():
            assert side in ['left','right'] and set(value)=={'kp','kd'} and 25.<=float(value['kp'])<=500. and 6.<=float(value['kd'])<=40.
            overrides[side+r'_wrist_[xyz]$']=dict(value)
        with open_dict(cfg.embodiment.config):cfg.embodiment.config.joint_overrides=overrides
    first=OmegaConf.to_container(cfg.curriculum.stages[0],resolve=True)
    schedule=([(0.,.9),(0.,.95),(0.,1.)] if stage=='prefix' else [(0.,.75),(0.,.8),(0.,.9)])
    stages=[]
    for voc,probability in schedule:
        value=json.loads(json.dumps(first));value['voc_scale']=voc;value['reset_to_first_frame_probability']=probability
        value['objective_weights']['hand_keypoints']=1.;value['objective_weights']['hand_joint_pos']=.5
        stages.append(value)
    cfg.curriculum.stages=stages
    cfg.curriculum.thresholds=([backward_stage*STAGE_BUDGET+RETRAIN_OFFSET+1_000_448,backward_stage*STAGE_BUDGET+RETRAIN_OFFSET+3_000_320,budget] if stage=='prefix' else
                              [PREFIX_BUDGET+5_000_192,PREFIX_BUDGET+20_000_768,TOTAL_BUDGET])
    configuration=OmegaConf.to_container(cfg,resolve=True)
    (ROOT/f'research/backward-grasp-{stage}-{backward_stage}-training-config.json').write_text(json.dumps(configuration,indent=2))
    (ROOT/'coherent-training-config.json').write_text(json.dumps(configuration,indent=2))
    (ROOT/'coherent-training-provenance.json').write_text(json.dumps({
        'episode':episode['sequence'],'engine':'Newton/FlashCHORD','algorithm':'FlashSAC',
        'initialization':'Actor learns holding, earlier pickup starts, then full motion. Every new phase disables object forces and reset settling; historical warmstarts are recorded separately.',
        'training_phase':stage,'prior_counter':0,'additional_training_step_budget':TOTAL_BUDGET,
        'prefix_training_steps':PREFIX_BUDGET,'full_motion_additional_steps':TOTAL_BUDGET-PREFIX_BUDGET,
        'backward_stage':backward_stage if stage=='prefix' else None,'training_start_source_frame':source_start,
        'adaptive_single_prefix':bool(episode.get('adaptive_single_prefix')),
        'reset_settling_steps':0,'training_object_force_limit_N':0.,'training_object_torque_limit_Nm':0.,
        'historical_supported_warmstart_environment_steps':RETRAIN_OFFSET,
        'training_finger_reset_initialization':'half exact-reference, half uniformly scaled 0..1; native inference remains exact-reference',
        'adaptive_boundary_reset_overlap':{'boundary_fraction':.7,'two_source_frames_earlier_fraction':.2,'original_frame_zero_fraction':.1},
        'adaptive_training_reset':adaptive,'evaluation_start_source_frame':0,
        'prefix_end_source_frame':prefix_end,'prefix_control_end_exclusive':prefix_control_end,'final_evaluation_uses_full_episode':True,
        'world_count':1024,'seed':44,'baseline_run':episode['baseline_run'],
        'baseline_inputs_copied_without_pose_or_geometry_edits':not bool(episode.get('physical_grasp_refinement')),
        'source_object_poses_and_camera_unchanged':True,
        'physical_hand_grasp_reference_refinement':episode.get('physical_grasp_refinement'),
        'wrist_motor_profile':episode.get('wrist_motor_profile'),
        'wrist_side_motor_profiles':episode.get('wrist_side_motor_profiles',{}),
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
        assert configuration['task']['motion_start_frame']==start_control
        assert configuration['env']['reset']['voc_decay_steps']==0
        assert configuration['env']['voc']['max_force']==0. and configuration['env']['voc']['max_torque']==0.
        assert configuration['env']['reset']['reset_finger_openness']==1.
        print('Validated grasp curriculum stage',stage,'without starting policy training.',flush=True);return
    previous.write_text(json.dumps(checksums,indent=2))
    try:native['main'].__wrapped__(cfg)
    except TrainingValidated as decision:
        print('Full-motion training stopped after repeated independent strong validation:',decision,flush=True)
        record=json.loads((ROOT/'research/training-early-stop.json').read_text())
        path=ROOT/'coherent-training-provenance.json';provenance=json.loads(path.read_text())
        provenance.update(planned_additional_training_step_budget=TOTAL_BUDGET,
            additional_training_step_budget=record['actual_environment_steps'],
            actual_training_environment_steps=record['actual_environment_steps'],
            validated_early_stop=True,training_stop_decision='research/training-early-stop.json')
        path.write_text(json.dumps(provenance,indent=2))


main()
if not validate:
    stopped=(ROOT/'research/training-early-stop.json').exists() and stage=='full'
    (folder/'TRAINING_COMPLETED').write_text('Full-motion experiment ended after repeated strong independent validation. See research/training-early-stop.json.\n' if stopped else f'Grasp curriculum {stage} training completed.\n')
    if stage=='prefix':
        if backward_stage==3:
            (prefix_root/'TRAINING_COMPLETED').write_text(
                'Continuous adaptive opening curriculum completed; fixed stages0..2 were not run.\n'
                if episode.get('adaptive_single_prefix') else 'All four backward pickup stages completed.\n')
            os.environ['V2D_GRASP_STAGE']='full'
        else:os.environ['V2D_GRASP_STAGE']='prefix'
        os.environ.pop('V2D_BACKWARD_STAGE',None)
        os.execv(sys.executable,[sys.executable,'-u',str(ROOT/'scripts/train_coherent_actor_transfer.py')])
