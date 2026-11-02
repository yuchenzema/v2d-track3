"""Train once and record one genuine official rollout; quality stays diagnostic."""
import json
from pathlib import Path
import shutil
import numpy as np
from improvement_common import BASE,save,utc,digest


def install_source(root,source):
    from own_geometry import validate
    root,source=Path(root),Path(source)
    proof=validate(source);cfg=json.loads((root/'episode-config.json').read_text())
    original=json.loads((source/'episode-config.json').read_text())
    assert cfg.get('fast_submission') is True and cfg.get('own_geometry_required') is True
    assert not cfg.get('actor_transfer'),'Scan-trained actors cannot initialize this cohort'
    # Retargeting/evaluation needs the estimated motion and support metadata,
    # not duplicate RGB/depth caches. Keep raw source inputs in their original
    # independently hashed frontend (and outside Docker training writes).
    for name in ['camera_intrinsics.json','hand_measurements.npz','table-camera-calibration.json',
                 'table-camera-calibration.npz','scene-reconstruction.json']:
        src=source/'reconstruction'/name;dst=root/'reconstruction'/name
        if src.is_file() and not dst.exists():shutil.copy2(src,dst)
    assets=root/'reconstruction/mesh_priors';assets.mkdir(exist_ok=True)
    for path in (source/'reconstruction/mesh_priors').iterdir():
        if path.is_file():shutil.copy2(path,assets/path.name)
    shutil.copy2(source/'research/own-geometry-provenance.json',root/'research/own-geometry-provenance.json')
    trajectory=source/'reconstruction/scene_motion_physical.npz'
    with np.load(trajectory,allow_pickle=False) as motion:
        assert motion['object_to_world'].shape==(original['frames'],len(original['objects']),4,4)
        for key in ['object_to_world','camera_to_world','hand_joint_positions']:
            assert np.isfinite(motion[key]).all(),('Nonfinite reconstructed input',key)
    target=root/'reconstruction/scene_motion_physical.npz'
    if (root/'coherent-input-checksums.json').exists():
        assert digest(target)==digest(trajectory),'Frozen training reference changed'
    else:shutil.copy2(trajectory,target)
    cfg.update(own_geometry_required=True,dataset_split='evaluation',objects=original['objects'],
        frames=original['frames'],fps=original['fps'],training_reference='track3_coherent_variant',
        fast_source_run=str(source),official_recorded_only=True,post_training_grasp_acceptance_required=False,
        require_nonpenetrating_reference_before_training=False,grasp_search_require_nonpenetrating_initial_pose=False,
        requires_strict_source_match_for_publication=False,requires_no_penetration_for_publication=False,
        own_training_continuation_allowed=False,strict_completion_granted=False)
    save(root/'episode-config.json',cfg)
    validate(root)
    save(root/'research/fast-source-provenance.json',dict(updated_utc=utc(),
        source=str(source),trajectory_sha256=digest(trajectory),own_geometry_proof_sha256=digest(source/'research/own-geometry-provenance.json'),
        input_reference_estimated=True,strict_alignment_verified=False,quality_gates_are_diagnostic=True,
        public_object_scans_used=False,source_clock_preserved=True))
    return cfg


def training_script(episode,seed):
    """Use the existing fresh learner with the same zero-assistance source clock."""
    text=(BASE/'fresh_improvement_training.py').read_text()
    text=text.replace('scene.world_count=1024','scene.world_count=512')
    text=text.replace('training.compute_dtype=bfloat16','training.compute_dtype=float32')
    text=text.replace("'sim.njmax=1536'","'sim.nconmax=2048','sim.njmax=4096'")
    text=text.replace("'reset.seed=43'",f"'reset.seed={seed}'")
    text=text.replace("'world_count': 1024","'world_count': 512").replace("'seed': 43",f"'seed': {seed}")
    text=text.replace("install_training_fingers(native['main'].__wrapped__.__globals__)","install_training_fingers(native['main'].__wrapped__.__globals__,focus=episode.get('training_focus'))")
    # Native supported scene option: contacts remain enabled, but the own mesh
    # uses its single convex hull rather than expensive concavity decomposition.
    text=text.replace("    validated=bool(episode.get('validated_video_pipeline'))",
        "    validated=bool(episode.get('validated_video_pipeline'))\n    if episode.get('fast_submission'):cfg.scene.decompose_objects=False")
    text=text.replace("'baseline_inputs_copied_without_pose_or_geometry_edits': True",
        "'baseline_inputs_copied_without_pose_or_geometry_edits': False, 'own_geometry_source': episode.get('fast_source_run'), 'collision_approximation': 'native single convex hull of the reconstructed mesh; real contacts enabled'")
    return text


def reset_untrained_setup(root):
    """Only a setup with zero learning/checkpoints may adopt faster physics."""
    root=Path(root);manifest=root/'coherent-input-checksums.json'
    if not manifest.exists():return False
    config=json.loads((root/'coherent-training-config.json').read_text())
    if config.get('scene',{}).get('decompose_objects') is False:return False
    folder=root/'flash_chord_coherent_actor_transfer'
    if list(folder.glob('*.safetensors')) or (folder/'TRAINING_COMPLETED').exists():return False
    metrics=folder/'metrics.jsonl'
    if metrics.exists():
        for line in metrics.read_text().splitlines():
            if json.loads(line).get('environment_steps',0)>0:return False
    for name,expected in json.loads(manifest.read_text()).items():
        assert digest(root/name)==expected,'Untrained reference changed before physics setup'
    history=root/'research/pretraining-setup-history'/digest(manifest)[:16];history.mkdir(parents=True,exist_ok=True)
    for name in ['coherent-input-checksums.json','coherent-training-config.json','coherent-training-provenance.json']:
        path=root/name
        if path.exists():path.rename(history/name)
    shutil.copy2(root/'scripts/train_coherent_actor_transfer.py',history/'train_coherent_actor_transfer.py')
    save(history/'reason.json',dict(updated_utc=utc(),actual_training_steps=0,
        checkpoint_existed=False,reason='Replace expensive pretraining CoACD setup with supported native convex-hull contacts'))
    return True


def run(root,job,make_container,status):
    from batch_generic_retarget import install
    from batch_native import docker,_docker,launch_training
    from official_policy_suite import evaluate_candidate,verify_trajectory
    root=Path(root);cfg=install_source(root,job['fast_source_run']);ep=cfg['episode']
    reset_untrained_setup(root)
    status(ep,'Fast submission: retarget the estimated source once; quality is diagnostic')
    if not (root/'coherent-input-checksums.json').exists():install(root)
    for helper in ['legacy_source_frame_playback.py','actor_transfer_training.py','run_geometry_cpu.py',
                   'training_numeric_failure_guard.py','training_finger_reset.py','robot_wrist_calibration.py']:
        target=root/'scripts'/helper
        if not target.exists():shutil.copy2(BASE/helper,target)
    native=make_container(root,ep,'newton')
    retarget=make_container(root,ep,'retarget')
    for script,variant in [('retarget_hands','track3_video'),('coherent_grasp_variant','track3_coherent_variant')]:
        output=root/f"human_motion_data/{variant}/processed/sequence_id={cfg['sequence']}/robot_name=sharpa_wave/motion.parquet"
        if not output.is_file():
            status(ep,'Fast submission: '+script)
            docker(['exec',retarget,'/isaac-sim/python.sh','-u',f'/episode/scripts/{script}.py'],root/f'logs/fast-{script}.log')
    script=root/'scripts/train_coherent_actor_transfer.py'
    if not (root/'coherent-input-checksums.json').exists():
        script.write_text(training_script(cfg,20261102+ep+997*job['generation']))
        _docker(['exec',native,'env','CUDA_VISIBLE_DEVICES=','JAX_PLATFORMS=cpu','V2D_VALIDATE_CONFIG_ONLY=1',
            'python','-u','/episode/scripts/run_geometry_cpu.py','/episode/scripts/train_coherent_actor_transfer.py'],root/'logs/fast-training-config-validation.log')
    if not (root/'flash_chord_coherent_actor_transfer/TRAINING_COMPLETED').is_file():
        status(ep,f"Fast submission: train one policy to {cfg['training_budget']:,} actual environment steps")
        save(root/'research/improvement-status.json',dict(status='training',episode=ep,updated_utc=utc()))
        launch_training(native,root)
    policies=list((root/'flash_chord_coherent_actor_transfer').glob('policy_*.safetensors'))
    if not policies:raise RuntimeError('No trained policy checkpoint; no rollout can be submitted')
    checkpoint=max(policies,key=lambda p:int(p.stem.rsplit('_',1)[1]))
    status(ep,'Fast submission: one official full-source policy rollout, starting at frame zero')
    save(root/'research/improvement-status.json',dict(status='evaluating',episode=ep,updated_utc=utc()))
    suite=evaluate_candidate(root,checkpoint,native)
    trajectory=Path(suite['recorded_trajectory']);states=Path(suite['recorded_body_states']);sha=digest(checkpoint)
    provenance=verify_trajectory(trajectory,sha,suite['official_entrypoint_sha256'],1,ep)
    assert int(provenance['flash_chord.step_count'])==cfg['frames']
    with np.load(states,allow_pickle=False) as record:
        assert bool(record['policy_inference']) and str(record['checkpoint_sha256'])==sha
        assert np.all(record['object_assistance']==0.) and len(record['body_poses'])==cfg['frames']
    actual=suite['attempts']['recorded']['actual_training_steps'];assert actual>0
    selected=dict(checkpoint=str(checkpoint),recorded_states=str(states.relative_to(root)),
        official_recorded_state_sha256=digest(states),official_trajectory_sha256=digest(trajectory),
        source_quality_is_diagnostic=True)
    save(root/'final-controller-selection.json',dict(selected=selected))
    result=dict(status='ready_for_submission',episode=ep,run=str(root),updated_utc=utc(),
        checkpoint=str(checkpoint),checkpoint_sha256=sha,actual_training_steps=actual,
        trajectory=str(trajectory),trajectory_sha256=digest(trajectory),states_sha256=digest(states),
        complete_source_frame_count=cfg['frames'],public_object_scans_used=False,
        frozen_input_manifest_sha256=digest(root/'coherent-input-checksums.json'),
        policy_inference=True,object_assistance=0.,strict_completion_granted=False,
        quality_warnings=['Strict motion/contact/grasp acceptance was not required under the user complete-first instruction.'],
        local_reference_metrics=suite['attempts']['recorded']['metrics'],local_metrics_are_not_kaggle_scores=True)
    save(root/'research/fast-submission-result.json',result)
    save(root/'research/improvement-status.json',dict(result,status='completed_for_submission'))
    status(ep,'Fast submission: trained policy and official trajectory ready; strict completion not claimed')
    return result
