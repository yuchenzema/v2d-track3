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


def eligible(root):
    """Recheck evidence for the exact chosen physical rollout and checkpoint."""
    from own_geometry import validate
    root=Path(root);cfg=read(root/'episode-config.json')
    if not cfg.get('own_geometry_required'):raise ValueError('Historical scan-derived run')
    if cfg.get('actor_transfer'):raise ValueError('Historical actor lineage')
    if cfg.get('training_initial_counter')!=0:raise ValueError('Fresh own-geometry training origin unverified')
    validate(root)
    if not (root/'flash_chord_coherent_actor_transfer/TRAINING_COMPLETED').is_file():raise ValueError('Training incomplete')
    from fast_submission_policy import enabled as fast_submission_enabled
    if cfg.get('fast_submission') and fast_submission_enabled():
        return eligible_fast(root,cfg)
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


def eligible_fast(root,cfg):
    """Submission integrity is mandatory; source/contact success is diagnostic."""
    import numpy as np
    from official_policy_suite import verify_trajectory
    result=read(root/'research/fast-submission-result.json')
    if result.get('status')!='ready_for_submission':raise ValueError('Fast policy rollout is not complete')
    if result.get('public_object_scans_used') is not False:raise ValueError('Unverified reconstruction lineage')
    checkpoint=Path(result['checkpoint']);sha=digest(checkpoint)
    if sha!=result['checkpoint_sha256']:raise ValueError('Trained checkpoint changed')
    suite=read(root/f'evaluation/official/{sha[:16]}/suite.json')
    if suite.get('status')!='official_evaluation_complete' or suite.get('checkpoint_sha256')!=sha:
        raise ValueError('Official policy evaluation is missing or stale')
    steps=suite.get('attempts',{}).get('recorded',{}).get('actual_training_steps',0)
    if not isinstance(steps,(int,float)) or steps<=0:raise ValueError('Actual trained policy steps missing')
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
        fast_result_sha256=digest(root/'research/fast-submission-result.json'),
        quality_acceptance_required=False,strict_completion_granted=False,
        meshes={name:str(root/f'reconstruction/mesh_priors/{name}.glb') for name in cfg['objects']})


def collect(focus):
    ready={};blocked={}
    for ep in focus:
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
    destination.mkdir(parents=True,exist_ok=True)
    with (destination/'upload.lock').open('a') as lock:
        fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
        receipt=destination/'kaggle-receipt.json'
        if receipt.is_file():
            # Reuse the IDs. Administrative invalidation or processing errors
            # must never automatically consume quota by retrying the upload.
            status.update(status='receipt_already_exists',receipt=str(receipt));save(out/'own-submission-status.json',status)
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
            if eligible(item['run'])!=item:raise ValueError('Own cohort changed during preflight')
        if read(BATCH/'user-pause.json').get('status')=='paused_by_user':return False
        subprocess.run(['git','push','origin','HEAD:main'],cwd=BASE,check=True,capture_output=True,text=True)
        status.update(status='uploading',file=str(package),sha256=digest(package));save(out/'own-submission-status.json',status)
        with (destination/'upload.log').open('a') as log:
            subprocess.run([str(PY),str(KIT/'submit/v2d_submit.py'),str(package),'--track','track_3',
                '-m',DESCRIPTION,
                '--wait','--receipt',str(receipt)],cwd=KIT,stdout=log,stderr=subprocess.STDOUT,check=True)
        status.update(status='submitted',receipt=str(receipt),updated_utc=utc());save(out/'own-submission-status.json',status)
    return False
