"""Pack and upload a verified own-reconstruction cohort in the existing controller.

No rollout is replaced with a target. No public scan, historical actor or stale
receipt can make a cohort eligible. Exact-match completion remains a separate
claim from submitting measured results. In complete-first mode, unsuccessful
motion/contact/grasp checks are diagnostic and never silently marked passed.
"""
import fcntl
import json
from pathlib import Path
import shutil
import subprocess
from improvement_common import BASE, BATCH, digest, save, utc
from rollout_collision_validation import collision_verified

PY=BASE/'.venv-submission/bin/python'
KIT=BASE/'submission_kit_20261102'
DESCRIPTION='Evaluation-only reconstructed geometry and actual trained-policy rollouts; no supplied scans'


def read(path):
    try:return json.loads(Path(path).read_text())
    except (OSError,ValueError):return {}


def receipt_states(receipt):
    pending=[];failed=[]
    for metric,row in receipt.get('submissions',{}).items():
        if row.get('state') in {'ERROR','UPLOAD_FAILED'} or not row.get('ref'):
            failed.append(metric)
        elif row.get('state')!='COMPLETE':pending.append(metric)
    return dict(pending=pending,failed=failed)


def attach_failed_upload_recovery(parent_path,recovery_path):
    """Attach only newly accepted IDs for a definite failed upload of this file."""
    parent_path=Path(parent_path);recovery_path=Path(recovery_path)
    parent=read(parent_path);recovery=read(recovery_path)
    assert parent['track']==recovery['track'],'Recovery track differs'
    assert Path(parent['file']).resolve()==Path(recovery['file']).resolve(),'Recovery file differs'
    replacements={}
    for metric,row in recovery['submissions'].items():
        old=parent['submissions'][metric]
        assert old['slug']==row['slug'],'Recovery competition differs'
        assert row.get('ref') and row.get('state') not in {'UPLOAD_FAILED','ERROR'},'Recovery was not accepted'
        if old.get('ref')==row['ref']:continue
        assert old.get('state')=='UPLOAD_FAILED' and not old.get('ref'),'Existing accepted IDs must be retained'
        replacements[metric]=dict(row,recovery_receipt=str(recovery_path.resolve()))
    if replacements:
        backup=parent_path.with_name(parent_path.stem+'-before-upload-recovery.json')
        if not backup.exists():save(backup,parent)
        parent['submissions'].update(replacements);save(parent_path,parent)
    return sorted(replacements)


def selections(out):
    """Retain a measured whole-episode choice over its known regressing run."""
    out=Path(out);selection=read(out/'quality-selection.json').get('episodes',{})
    for ep,row in read(out/'quality-kaggle-selection-overrides.json').get('episodes',{}).items():
        if selection.get(ep,{}).get('run')==row.get('rejected_run'):
            if not row.get('trial_receipt') or not row.get('baseline_receipt'):continue
            from kaggle_progress import compare_receipts
            verified=compare_receipts(row['baseline_receipt'],row['trial_receipt'])
            if not verified.get('all_six_strictly_improved'):continue
            selection[ep]=row['selection']
    return selection


def completed_snapshot_before_continuation(root,cfg,result_override=None):
    """A later budget does not erase an already recorded, completed model."""
    if not (cfg.get('quality_training') and cfg.get('own_training_continuation_allowed')):return False
    result=read(result_override or root/'research/fast-submission-result.json')
    sha=result.get('checkpoint_sha256','');steps=result.get('actual_training_steps',0)
    archived=read(root/f'research/quality-trained-candidates/{sha[:16]}.json')
    keys=['checkpoint_sha256','actual_training_steps','trajectory_sha256','states_sha256','frozen_input_manifest_sha256']
    if not sha or steps<=0 or any(result.get(k)!=archived.get(k) for k in keys):return False
    from safetensors import safe_open
    checkpoint=Path(result['checkpoint'])
    if checkpoint.parent.resolve()!=(root/'flash_chord_coherent_actor_transfer').resolve():return False
    if checkpoint.name!=f'policy_{steps}.safetensors' or digest(checkpoint)!=sha:return False
    with safe_open(checkpoint,framework='numpy') as file:
        if int(file.metadata().get('user.environment_steps',0))!=steps:return False
    for path in root.glob('research/training-continuation-*.json'):
        proof=read(path);state=Path(proof.get('from_checkpoint',''))
        if (proof.get('resume_environment_steps',0)>=steps and proof.get('frozen_input_hashes_verified') is True
                and proof.get('optimizer_restore_required') is True and state.is_file()
                and state.parent.resolve()==checkpoint.parent.resolve()
                and digest(state)==proof.get('from_checkpoint_sha256')):return True
    return False


def eligible(root,result_override=None):
    """Recheck evidence for the exact chosen physical rollout and checkpoint."""
    from own_geometry import validate
    root=Path(root);cfg=read(root/'episode-config.json')
    if not cfg.get('own_geometry_required'):raise ValueError('Historical scan-derived run')
    if cfg.get('actor_transfer'):raise ValueError('Historical actor lineage')
    if cfg.get('training_initial_counter')!=0:raise ValueError('Fresh own-geometry training origin unverified')
    validate(root)
    if not (root/'flash_chord_coherent_actor_transfer/TRAINING_COMPLETED').is_file():
        if not completed_snapshot_before_continuation(root,cfg,result_override):raise ValueError('Training incomplete')
    from fast_submission_policy import enabled as fast_submission_enabled
    if cfg.get('fast_submission') and fast_submission_enabled():
        return eligible_fast(root,cfg,result_override)
    strict=read(root/'research/strict-human-match.json')
    if not (strict.get('all_frames_checked') is True and strict.get('passes_strict_local_checks') is True
            and strict.get('failed_frame_count')==0 and strict.get('unobservable_frame_count')==0
            and strict.get('source_time_warping_used') is False):
        raise ValueError('Strict local source-motion checks remain unresolved')
    if not collision_verified(strict.get('collision_validation') or {},strict.get('states_sha256'),strict.get('version'),cfg['frames']):
        raise ValueError('Recorded rollout contact checks remain unresolved')
    selection=read(root/'final-controller-selection.json').get('selected',{})
    checkpoint=Path(selection.get('checkpoint',''))
    if str(checkpoint).startswith('/episode/'):checkpoint=root/str(checkpoint).removeprefix('/episode/')
    elif not checkpoint.is_absolute():checkpoint=root/checkpoint
    sha=digest(checkpoint);suite=read(root/f'evaluation/official/{sha[:16]}/suite.json')
    if suite.get('status')!='official_evaluation_complete' or suite.get('checkpoint_sha256')!=sha:
        raise ValueError('Official policy replay is incomplete or stale')
    if int(suite.get('attempts',{}).get('recorded',{}).get('actual_training_steps',0))<=0:
        raise ValueError('No actual policy training steps recorded')
    grasp=strict.get('post_training_grasp_acceptance') or {}
    if cfg.get('post_training_grasp_acceptance_required') and not (grasp.get('all_grasps_passed') is True
            and grasp.get('checkpoint_sha256')==sha and grasp.get('policy_inference') is True):
        raise ValueError('Policy grasp acceptance remains unresolved')
    trajectory=Path(suite['recorded_trajectory']);states=Path(suite['recorded_body_states'])
    if digest(states)!=strict.get('states_sha256'):raise ValueError('Strict checks belong to another recorded attempt')
    from official_policy_suite import verify_trajectory
    provenance=verify_trajectory(trajectory,sha,suite['official_entrypoint_sha256'],1,cfg['episode'])
    if int(provenance['flash_chord.step_count'])!=cfg['frames']:raise ValueError('Official source clock mismatch')
    return dict(run=str(root),episode=cfg['episode'],checkpoint=str(checkpoint),checkpoint_sha256=sha,
        trajectory=str(trajectory),trajectory_sha256=digest(trajectory),official_report=str(trajectory.with_suffix('.json')),
        strict_report_sha256=digest(root/'research/strict-human-match.json'),
        meshes={name:str(root/f'reconstruction/mesh_priors/{name}.glb') for name in cfg['objects']})


def eligible_fast(root,cfg,result_override=None):
    """Submission integrity is mandatory; source/contact success is diagnostic."""
    import numpy as np
    from official_policy_suite import verify_trajectory
    result_path=Path(result_override) if result_override else root/'research/fast-submission-result.json'
    result=read(result_path)
    if result.get('status')!='ready_for_submission':raise ValueError('Fast policy rollout is not complete')
    if result.get('public_object_scans_used') is not False:raise ValueError('Unverified reconstruction lineage')
    checkpoint=Path(result['checkpoint']);sha=digest(checkpoint)
    if sha!=result['checkpoint_sha256']:raise ValueError('Trained checkpoint changed')
    if result_override and result.get('control_calibration'):
        from safetensors import safe_open
        from safetensors.numpy import load_file
        from own_policy_calibration import weights_digest
        proof=read(checkpoint.with_suffix('.derivation.json'));parent=Path(proof['parent'])
        if digest(parent)!=proof['parent_sha256']:raise ValueError('Calibration parent changed')
        original=read(root/'research/fast-submission-result.json')
        if proof['parent_sha256']!=original['checkpoint_sha256']:raise ValueError('Calibration must originate from this own-geometry trained policy')
        if weights_digest(load_file(parent))!=weights_digest(load_file(checkpoint)):raise ValueError('Calibration edited learned weights')
        with safe_open(checkpoint,framework='numpy') as file:
            metadata=file.metadata()
            if int(metadata['user.environment_steps'])!=original['actual_training_steps']:raise ValueError('Calibration falsified training steps')
    elif result_override:
        # An archived, unmodified trained checkpoint is also a valid complete
        # policy choice. It must be a real learner output from this run.
        from safetensors import safe_open
        if checkpoint.resolve().parent!=(root/'flash_chord_coherent_actor_transfer').resolve():raise ValueError('Whole-checkpoint override is outside the trained policy folder')
        steps=int(result['actual_training_steps'])
        if checkpoint.name!=f'policy_{steps}.safetensors':raise ValueError('Whole-checkpoint filename and actual steps differ')
        with safe_open(checkpoint,framework='numpy') as file:
            metadata=file.metadata()
        if int(metadata['user.environment_steps'])!=steps or metadata.get('user.control_calibration_profile_json'):raise ValueError('Whole-checkpoint training provenance differs')
    suite=read(root/f'evaluation/official/{sha[:16]}/suite.json')
    if suite.get('status')!='official_evaluation_complete' or suite.get('checkpoint_sha256')!=sha:
        raise ValueError('Official policy evaluation is missing or stale')
    steps=suite.get('attempts',{}).get('recorded',{}).get('actual_training_steps',0)
    if not isinstance(steps,(int,float)) or steps<=0:raise ValueError('Actual trained policy steps missing')
    if result['actual_training_steps']!=steps:raise ValueError('Policy result and official replay counters differ')
    trajectory=Path(suite['recorded_trajectory']);states=Path(suite['recorded_body_states'])
    if digest(trajectory)!=result['trajectory_sha256'] or digest(states)!=result['states_sha256']:
        raise ValueError('Official recorded artifacts changed')
    provenance=verify_trajectory(trajectory,sha,suite['official_entrypoint_sha256'],1,cfg['episode'])
    if int(provenance['flash_chord.step_count'])!=cfg['frames']:raise ValueError('Incomplete source frame grid')
    with np.load(states,allow_pickle=False) as data:
        if not bool(data['policy_inference']) or str(data['checkpoint_sha256'])!=sha:
            raise ValueError('Recording is not inference from this policy')
        if not np.all(data['object_assistance']==0.):raise ValueError('Recording used object force assistance')
        if len(data['body_poses'])!=cfg['frames'] or not np.isfinite(data['body_poses']).all():
            raise ValueError('Incomplete/nonfinite policy states')
        if not np.array_equal(data['reference_frames'],np.arange(cfg['frames'])):
            raise ValueError('Recorded source clock/frame order changed')
    frozen=root/'coherent-input-checksums.json'
    if not frozen.is_file() or not read(frozen):raise ValueError('Frozen training input manifest missing')
    if digest(frozen)!=result.get('frozen_input_manifest_sha256'):raise ValueError('Training input manifest changed')
    for name,expected in read(frozen).items():
        if digest(root/name)!=expected:raise ValueError('Frozen own-policy training input changed')
    return dict(run=str(root),episode=cfg['episode'],checkpoint=str(checkpoint),checkpoint_sha256=sha,
        trajectory=str(trajectory),trajectory_sha256=digest(trajectory),official_report=str(trajectory.with_suffix('.json')),
        fast_result_sha256=digest(result_path),
        **({'result_override':str(result_path)} if result_override else {}),
        quality_acceptance_required=False,strict_completion_granted=False,
        meshes={name:str(root/f'reconstruction/mesh_priors/{name}.glb') for name in cfg['objects']})


def collect(focus):
    ready={};blocked={}
    selection=selections(BATCH/'focused_episodes_20261028')
    for ep in focus:
        if str(ep) in selection:
            row=selection[str(ep)]
            try:ready[str(ep)]=eligible(row['run'],row.get('result_override'));continue
            except (ValueError,AssertionError,OSError,KeyError) as error:blocked[str(ep)]=[dict(run=row['run'],reason=str(error))];continue
        roots=sorted((BASE/'runs').glob(f'episode_{ep:06d}_newton_human_match_20261023_g*'),reverse=True)
        for root in roots:
            if not read(root/'episode-config.json').get('own_geometry_required'):continue
            try:ready[str(ep)]=eligible(root);break
            except (ValueError,AssertionError,OSError,KeyError) as error:blocked.setdefault(str(ep),[]).append(dict(run=str(root),reason=str(error)))
        if str(ep) not in ready and str(ep) not in blocked:blocked[str(ep)]=[dict(reason='No own-geometry policy has completed yet')]
    return ready,blocked


def run(out,focus):
    from own_geometry import required
    if not required(out):return False
    policy=read(out/'own-geometry-policy.json')
    defer_scores=read(out/'quality-improvement-policy.json').get('enabled',False)
    if read(BATCH/'user-pause.json').get('status')=='paused_by_user':return False
    if read(out/'controller-reload-request.json').get('pending'):return False
    ready,blocked=collect(focus)
    status=dict(updated_utc=utc(),ready_episodes=sorted(map(int,ready)),required=20,
        blocked=blocked,strict_completion_granted=False,automatic_upload_authorized=policy.get('automatic_upload_after_full_verification') is True)
    previous=read(out/'own-submission-status.json')
    if previous.get('receipt'):status['receipt']=previous['receipt']
    save(out/'own-submission-status.json',status)
    if len(ready)!=len(focus) or len(focus)!=20:return False
    if policy.get('automatic_upload_after_full_verification') is not True:return False
    import hashlib
    signature=hashlib.sha256(json.dumps(ready,sort_keys=True).encode()).hexdigest()
    destination=BATCH/'own_geometry_submissions'/signature[:16]
    trials=list((BATCH/'quality_ablations_20261103').glob('*/kaggle-receipt.json'))
    receipts=([Path(previous['receipt'])] if previous.get('receipt') else [])+trials
    metrics={'add_auc','spider_sr','maniptrans_sr','rpe_cm','mppe_cm','cd_o_cm'}
    for prior in receipts:
        if (prior.is_file() and set(read(prior).get('submissions',{}))==metrics
                and read(prior.parent/'cohort-provenance.json').get('episodes')==ready):
            destination=prior.parent;break
    destination.mkdir(parents=True,exist_ok=True)
    with (destination/'upload.lock').open('a') as lock:
        fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
        receipt=destination/'kaggle-receipt.json'
        if receipt.is_file():
            # Reuse the IDs. Administrative invalidation or processing errors
            # must never automatically consume quota by retrying the upload.
            states=receipt_states(read(receipt));pending=bool(states['pending'])
            if pending and not defer_scores:
                status.update(status='waiting_for_scores',receipt=str(receipt));save(out/'own-submission-status.json',status)
                with (destination/'upload.log').open('a') as log:
                    subprocess.run([str(PY),str(KIT/'submit/v2d_submit.py'),'--track','track_3',
                        '--poll-receipt',str(receipt)],cwd=KIT,stdout=log,stderr=subprocess.STDOUT,check=True)
            status.update(status='partial_upload_failed' if states['failed'] else 'uploaded_waiting_for_scores' if pending and defer_scores else 'receipt_already_exists',failed_metrics=states['failed'],receipt=str(receipt));save(out/'own-submission-status.json',status)
            return False
        results=destination/'results';meshes=destination/'meshes';results.mkdir(exist_ok=True);meshes.mkdir(exist_ok=True)
        commit=subprocess.check_output(['git','rev-parse','HEAD'],cwd=BASE,text=True).strip()
        if subprocess.check_output(['git','status','--porcelain','--untracked-files=no'],cwd=BASE,text=True).strip():
            raise RuntimeError('Commit producing code before packing/uploading the own-geometry cohort')
        for ep,item in ready.items():
            # Read the exact official single-world trajectory; never resample,
            # splice attempts or correct achieved object poses after replay.
            shutil.copy2(item['trajectory'],results/f'episode_{int(ep):06d}.parquet')
            shutil.copy2(item['official_report'],results/f'episode_{int(ep):06d}.json')
            folder=meshes/f'episode_{int(ep):06d}';folder.mkdir(exist_ok=True)
            for name,path in item['meshes'].items():shutil.copy2(path,folder/f'{name}.glb')
        save(destination/'cohort-provenance.json',dict(episodes=ready,producing_code_commit=commit,source_clock_preserved=True,public_scans_used=False))
        package=destination/'track3_submission.parquet'
        commands=[['tools/pack_submission.py','--track','track_3','--results',str(results),'--meshes',str(meshes),
            '--code-commit',f'https://github.com/yuchenzema/v2d-track3/commit/{commit}','-o',str(package)],
            ['tools/preflight.py','--track','track_3','--results',str(results)],
            ['submit/v2d_submit.py',str(package),'--track','track_3','-m',DESCRIPTION,'--dry-run']]
        status.update(status='packing_and_preflight',destination=str(destination));save(out/'own-submission-status.json',status)
        for index,arguments in enumerate(commands):
            if read(BATCH/'user-pause.json').get('status')=='paused_by_user':return False
            with (destination/f'preflight-{index}.log').open('w') as log:
                subprocess.run([str(PY),str(KIT/arguments[0]),*arguments[1:]],cwd=KIT,stdout=log,stderr=subprocess.STDOUT,check=True)
        # Recheck the scientific artifacts immediately before the irreversible
        # upload; a newly replaced mesh/checkpoint cannot inherit old checks.
        for item in ready.values():
            if eligible(item['run'],item.get('result_override'))!=item:raise ValueError('Own cohort changed during preflight')
        if read(BATCH/'user-pause.json').get('status')=='paused_by_user':return False
        subprocess.run(['git','push','origin','HEAD:main'],cwd=BASE,check=True,capture_output=True,text=True)
        status.update(status='uploading',file=str(package),sha256=digest(package),receipt=str(receipt));save(out/'own-submission-status.json',status)
        with (destination/'upload.log').open('a') as log:
            subprocess.run([str(PY),str(KIT/'submit/v2d_submit.py'),str(package),'--track','track_3',
                '-m',DESCRIPTION,
                *([] if defer_scores else ['--wait']),'--receipt',str(receipt)],cwd=KIT,stdout=log,stderr=subprocess.STDOUT,check=True)
        entries=read(receipt).get('submissions',{})
        pending=any(item.get('state') not in {'COMPLETE','ERROR'} for item in entries.values())
        status.update(status='uploaded_waiting_for_scores' if pending else 'submitted',receipt=str(receipt),updated_utc=utc());save(out/'own-submission-status.json',status)
    return False
