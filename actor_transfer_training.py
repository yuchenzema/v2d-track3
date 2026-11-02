"""Actor-only warm start with native policy-schema checks and fresh critics.

The original checkpoint is immutable. New checkpoints retain their own source
reference, unit source clock, motor/action configuration and training counter.
"""
import hashlib
import json
from pathlib import Path
import struct


def metadata(path):
    with Path(path).open('rb') as stream:
        length=struct.unpack('<Q',stream.read(8))[0]
        if not 0<length<=16*1024*1024:raise ValueError('Invalid checkpoint header length')
        return json.loads(stream.read(length))['__metadata__']


def recipe(episode,base):
    if episode!=38:return None
    checkpoint=Path(base)/'runs/episode_000038_newton_batch_20261015/flash_chord_coherent_actor_transfer/policy_90000384.safetensors'
    parent=Path(base)/'runs/episode_000038_newton_human_match_20261023_g007'
    selected_from=None
    try:
        strict=json.loads((parent/'research/strict-human-match.json').read_text())
        metrics=json.loads((parent/'evaluation/final-metrics.json').read_text())
        grasp=strict['post_training_grasp_acceptance'];candidate=parent/metrics['checkpoint'].removeprefix('/episode/')
        sha=hashlib.sha256(candidate.read_bytes()).hexdigest()
        moving=bool(grasp.get('grasps')) and all(g['expected_motion_m']>.015 and
            g['actual_motion_m']>=.75*g['expected_motion_m'] for g in grasp['grasps'])
        # Choose an entire measured actor, never a successful rollout segment.
        # Eligibility to initialize another experiment is not episode acceptance.
        if (strict.get('all_frames_checked') is True and grasp.get('policy_inference') is True
                and grasp.get('source_frames_checked')==strict.get('source_frames')
                and metrics['checkpoint_sha256']==sha==strict['checkpoint_sha256']==grasp['checkpoint_sha256']
                and metrics['metrics']['object_position_error_m']<.05 and moving):
            selected_from=dict(ancestor_checkpoint=str(checkpoint),
                ancestor_checkpoint_sha256=hashlib.sha256(checkpoint.read_bytes()).hexdigest(),
                ancestor_environment_steps=90000384,source_quality_is_local_not_kaggle=True,
                source_episode_strictly_passed=False)
            checkpoint=candidate
    except (OSError,KeyError,TypeError,ValueError):pass
    info=metadata(checkpoint)
    assert info['user.checkpoint_kind']=='policy' and info['user.algorithm']=='flash_sac'
    result=dict(source_checkpoint=str(checkpoint),source_checkpoint_sha256=hashlib.sha256(checkpoint.read_bytes()).hexdigest(),
        source_environment_steps=int(info['user.environment_steps']),
        components=['actor.params','actor.batch_stats'],fresh_critics=True,fresh_optimizer=True,
        preserve_source_actor_action_semantics=True,training_counter_starts_at_zero=True)
    if selected_from:result.update(selected_from)
    return result


def configure(config,description):
    from omegaconf import OmegaConf,open_dict
    saved=json.loads(metadata(description['checkpoint'])['user.resolved_config_json'])
    with open_dict(config):
        config.action=OmegaConf.create(saved['action'])
        config.training.network=OmegaConf.create(saved['training']['network'])


def install(runner,description,current_metadata,root):
    from flash_chord.training.checkpoint import read_checkpoint_metadata
    from flash_chord.training.checkpoint_metadata import checkpoint_policy_schema,validate_policy_schema
    from flash_chord.training.flash_sac.checkpoint import ActorInferenceState,load_actor_checkpoint
    checkpoint=Path(description['checkpoint'])
    actual=hashlib.sha256(checkpoint.read_bytes()).hexdigest()
    assert actual==description['source_checkpoint_sha256'],'Transfer actor changed'
    saved=read_checkpoint_metadata(checkpoint)
    validate_policy_schema(checkpoint_policy_schema(saved),checkpoint_policy_schema(current_metadata))
    original=runner.create_learner

    def create(*args,**kwargs):
        import jax
        learner,executables=original(*args,**kwargs)
        assert int(jax.device_get(learner.environment_steps))==0
        template=ActorInferenceState(params=learner.actor.params,batch_stats=learner.actor.batch_stats)
        actor,_=load_actor_checkpoint(checkpoint,template)
        learner=learner.replace(actor=learner.actor.replace(params=actor.params,batch_stats=actor.batch_stats))
        proof=dict(source_checkpoint_sha256=actual,source_environment_steps=description['source_environment_steps'],
            loaded_components=description['components'],native_policy_schema_validated=True,
            critics_and_target_critics_fresh=True,actor_optimizer_fresh=True,reward_normalizer_fresh=True,
            replay_fresh=True,new_environment_steps=int(jax.device_get(learner.environment_steps)),
            reference_or_object_state_edited=False)
        (Path(root)/'research/actor-transfer-initialization.json').write_text(json.dumps(proof,indent=2))
        return learner,executables

    runner.create_learner=create
    return original
