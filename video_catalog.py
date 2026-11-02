"""Resolve GUI video links to complete, hash-verified rollout releases.

Trials remain available in history; current links retain the best verified policy.
Partially rendered or failed experiments never replace a playable release.
"""
from pathlib import Path
from datetime import datetime
from urllib.parse import quote
import hashlib
import json
import math

BASE=Path(__file__).resolve().parent
RUNS=BASE/'runs'
BATCH=RUNS/'evaluation_batch_20261015'
OUT=BATCH/'improvements_20261019'
HASH_CACHE={}


def read(path):
    return json.loads(Path(path).read_text())


def verified(path,expected):
    path=Path(path).resolve()
    if not path.is_relative_to(RUNS.resolve()) or not path.is_file():
        raise ValueError(f'Unavailable release artifact: {path.name}')
    stat=path.stat();key=(str(path),stat.st_mtime_ns,stat.st_size,expected)
    if key not in HASH_CACHE:
        digest=hashlib.sha256(path.read_bytes()).hexdigest()
        if digest!=expected:raise ValueError(f'Video release hash mismatch: {path}')
        HASH_CACHE[key]=True
    return path


def url(path,version):
    return '/artifacts/'+quote(str(Path(path).relative_to(RUNS)),safe='/')+'?v='+version


def strict_release_matches(root,version,selected):
    """Fail closed when a strict audit is missing or belongs to another rollout."""
    try:
        gate=read(Path(root)/'research/strict-human-match.json')
        if not (gate.get('passes_strict_local_checks') and gate.get('version')==version and gate.get('states_sha256')):
            return False
        states=Path(selected['recorded_states'])
        if str(states).startswith('/episode/'):states=Path(root)/str(states).removeprefix('/episode/')
        elif not states.is_absolute():states=Path(root)/states
        verified(states,gate['states_sha256'])
        from rollout_collision_validation import collision_verified
        collision=gate.get('collision_validation') or read(Path(root)/'research/rollout-collision-validation.json')
        if not collision_verified(collision,gate['states_sha256'],version):return False
        return True
    except (OSError,ValueError,KeyError,TypeError):
        return False


def run_release(ep,root,kind,result=None):
    root=Path(root).resolve()
    if not root.is_relative_to(RUNS.resolve()):raise ValueError('Release is outside runs')
    cfg=read(root/'episode-config.json')
    if cfg['episode']!=ep:raise ValueError('Video release episode identity mismatch')
    report=read(root/'result-report.json');manifest=read(root/'deliverables/manifest.json')
    files={}
    for key,name in [('comparison','original_vs_policy.mp4'),('rollout','rollout_source_timed.mp4'),
                     ('report','report.json')]:
        entry=manifest[name];files[key]=verified(root/entry['path'],entry['sha256'])
    version=hashlib.sha256(json.dumps({k:manifest[n]['sha256'] for k,n in
        [('comparison','original_vs_policy.mp4'),('rollout','rollout_source_timed.mp4'),('report','report.json')]},sort_keys=True).encode()).hexdigest()[:16]
    if 'rollout_3d.mp4' in manifest:
        entry=manifest['rollout_3d.mp4'];files['scene']=verified(root/entry['path'],entry['sha256'])
    selected=read(root/'final-controller-selection.json')['selected']
    pickup_ok=True
    if cfg.get('requires_no_penetration_for_publication'):
        from rollout_collision_validation import collision_verified
        collision_path=root/'research/rollout-collision-validation.json'
        collision=read(collision_path) if collision_path.exists() else {}
        states=Path(selected['recorded_states'])
        if str(states).startswith('/episode/'):states=root/str(states).removeprefix('/episode/')
        elif not states.is_absolute():states=root/states
        pickup_ok=collision_verified(collision,hashlib.sha256(states.read_bytes()).hexdigest(),version,2*(cfg['frames']-1))
    if cfg.get('requires_strict_source_match_for_publication'):
        pickup_ok=pickup_ok and strict_release_matches(root,version,selected)
    if cfg.get('requires_brush_pickup_validation'):
        gate_path=root/'research/learned-policy-pickup-gate.json'
        gate=read(gate_path) if gate_path.exists() else {}
        pickup_ok=pickup_ok and bool(gate.get('passes') and gate.get('states_sha256'))
        if pickup_ok:
            try:verified(root/selected['recorded_states'],gate['states_sha256'])
            except (OSError,ValueError):pickup_ok=False
    return {'episode':ep,'kind':kind,'run':str(root),'version':version,
            'validated_utc':result.get('updated_utc') if result else None,
            'measured_improvement':bool(result and result.get('measured_improvement')) and pickup_ok,
            'eligible_for_latest':pickup_ok and ((cfg.get('training_curriculum')!='contact_finetune' and not cfg.get('requires_verified_improvement_for_publication')) or bool(result and result.get('measured_improvement'))),
            'label':'Original baseline' if kind=='baseline' else 'Improved policy' if result.get('measured_improvement') and pickup_ok else 'New trial · improvement not confirmed',
            'integrity_verified':True,'quality':report['quality_assessment'],
            'object_motion_quality':selected.get('object_motion_quality'),
            'checkpoint_sha256':selected.get('checkpoint_sha256'),
            'recorded_states':str(root/selected['recorded_states']),
            'source_mean_object_iou':selected.get('source_mean_object_iou'),
            'source_object_metrics':selected.get('source_metrics',{}).get('objects',{}),
            'native_object_tracking':selected.get('single_attempt_tracking',{}).get('per_body',{}),
            'links':{key:url(path,version) for key,path in files.items()},
            'selected':selected}


def controller_release(ep,result,result_path=None):
    rendered=result['rendered'];root=Path(result.get('evaluation_run') or rendered.get('evaluation_run') or RUNS/f'episode_{ep:06d}_newton_batch_20261015')
    files={}
    for entry in rendered['validated_videos']:
        path=verified(entry['path'],entry['sha256'])
        key={'original_vs_improved_policy.mp4':'comparison','policy_3d.mp4':'scene',
             'original_before_after.mp4':'before_after','rollout_source_timed.mp4':'rollout'}.get(path.name)
        if key:files[key]=path
    if 'comparison' not in files:raise ValueError('Improved controller comparison is missing')
    version=hashlib.sha256(json.dumps(rendered['validated_videos'],sort_keys=True).encode()).hexdigest()[:16]
    links={key:url(path,version) for key,path in files.items()}
    reproduction=None
    if rendered.get('reproduce_script'):
        verification=Path(rendered['reproduce_script']).with_name('reproduction-verification.json')
        if verification.is_file():
            reproduction=read(verification);links['reproduction']=url(verification,version)
    if result_path:links['report']=url(Path(result_path),version)
    diagnostic=result.get('release_type')=='diagnostic'
    training_complete=(root/'flash_chord_coherent_actor_transfer/TRAINING_COMPLETED').exists()
    quality=dict(result.get('quality_assessment') or {'all_internal_frame_checks_passed':False,
                       'completion_rate_against_video_estimate':result['native_128_world_evaluation']['metrics']['chord_sr']})
    if training_complete:quality['training_continues']=False
    return {'episode':ep,'kind':'diagnostic' if diagnostic else 'controller','run':str(root),'version':version,
            'validated_utc':result['updated_utc'],'measured_improvement':bool(result.get('measured_improvement')),
            'eligible_for_latest':not diagnostic,
            'label':result.get('label','Candidate under review') if diagnostic else 'Improved controller · bounded wrist feedback' if result['selected'].get('robot_tracking_feedback_profile') else 'Improved controller','integrity_verified':True,
            'training_complete':training_complete,'quality':quality,
            'object_motion_quality':result['selected'].get('object_motion_quality'),
            'checkpoint_sha256':result['selected'].get('checkpoint_sha256'),
            'recorded_states':str(root/result['selected']['recorded_states']),
            'source_mean_object_iou':result['selected'].get('source_mean_object_iou'),
            'source_object_metrics':result['selected'].get('source_metrics',{}).get('objects',{}),
            'native_object_tracking':result['selected'].get('single_attempt_tracking',{}).get('per_body',{}),
            'independent_reproduction':reproduction,
            'links':links,'selected':result['selected']}


def catalog(ep,baseline_root,include_selected=False):
    releases=[];issues=[]
    if ep in (1,5) and not (Path(baseline_root)/'episode-config.json').exists():
        try:
            data=read(BATCH/'all20_evaluation'/f'episode_{ep:06d}-release.json')
            selected=data.get('selected') or {}
            files={key:verified(value['path'],value['sha256']) for key,value in data['files'].items()}
            release={'episode':ep,'kind':'baseline','run':data['run'],'version':data['version'],
                     'validated_utc':data['validated_utc'],'measured_improvement':False,
                     'eligible_for_latest':True,'label':'Preserved preferred policy' if ep==1 else 'Completed episode5 policy',
                     'integrity_verified':True,'quality':data['quality'],
                     'object_motion_quality':selected.get('object_motion_quality'),
                     'checkpoint_sha256':data.get('checkpoint_sha256'),
                     'recorded_states':data.get('recorded_states'),
                     'source_mean_object_iou':selected.get('source_mean_object_iou'),
                     'source_object_metrics':selected.get('source_metrics',{}).get('objects',{}),
                     'native_object_tracking':selected.get('single_attempt_tracking',{}).get('per_body',{}),
                     'links':{key:url(path,data['version']) for key,path in files.items()}}
            release['selected']=data['selected']
            releases.append(release)
        except (OSError,ValueError,KeyError) as exc:
            issues.append(str(exc))
    if not releases:
        try:releases.append(run_release(ep,baseline_root,'baseline'))
        except (OSError,ValueError,KeyError) as exc:issues.append(str(exc))
    # Globbing phase result records also finds future optimization rounds.
    for directory in sorted(BATCH.glob('improvements_*')):
        for path in sorted(directory.glob(f'episode_{ep:06d}-*result.json')):
            try:
                result=read(path)
                if result.get('status')!='completed':continue
                if 'run' in result:
                    if not result.get('artifact_integrity',{}).get('integrity_passed'):continue
                    release=run_release(ep,result['run'],'experiment',result)
                elif result.get('rendered') and (result.get('measured_improvement') or result.get('release_type')=='diagnostic'):
                    if result.get('release_type')=='checkpoint' and not any(
                        Path(v['path']).name=='rollout_source_timed.mp4'
                        for v in result['rendered']['validated_videos']):continue
                    release=controller_release(ep,result,path)
                    if result.get('release_type')=='checkpoint':
                        release.update(kind='checkpoint',label='Verified checkpoint · training complete' if release['training_complete'] else 'Improved checkpoint · training continues')
                else:continue
                review_path=Path(release['run'])/'research/rejected-inference-references.json'
                if review_path.exists():
                    rejection=read(review_path);reference=release['selected'].get('inference_reference')
                    if reference in rejection.get('references',[]):
                        release.update(measured_improvement=False,eligible_for_latest=False,
                                       label='Diagnostic · source reconstruction rejected',source_review_rejection=rejection['reason'])
                if not any(r['version']==release['version'] for r in releases):releases.append(release)
            except (OSError,ValueError,KeyError) as exc:issues.append(f'{path.name}: {exc}')
    def timestamp(release):
        value=release.get('validated_utc')
        return datetime.fromisoformat(value.replace('Z','+00:00')).timestamp() if value else 0.
    releases.sort(key=timestamp)
    if not releases:return {'latest':None,'best':None,'baseline':None,'history':[],'issues':issues}
    original=next((r for r in releases if r['kind']=='baseline'),None)
    best=original or releases[0]
    from improvement_common import better
    from object_motion_quality import attach
    for release in releases:
        if release is best or not release['measured_improvement']:continue
        try:
            chosen=attach(release['run'],release['selected'])
            previous=attach(best['run'],best['selected'])
            cohort=float(release['quality']['completion_rate_against_video_estimate'])
            previous_cohort=float(best['quality']['completion_rate_against_video_estimate'])
            if previous is None:
                strict=read(Path(release['run'])/'research/strict-human-match.json')
                if strict.get('version')==release['version'] and strict.get('passes_strict_local_checks') and math.isfinite(cohort) and cohort>=previous_cohort-.0625:best=release
                continue
            if math.isfinite(cohort) and cohort>=previous_cohort-.0625 and better(chosen,previous):best=release
        except (OSError,ValueError,KeyError) as exc:issues.append(f'Best-policy comparison: {exc}')
    def public(release):
        return {key:value for key,value in release.items() if include_selected or key!='selected'} if release else None
    # Once a policy is promoted, current links retain that accepted improvement.
    # Subsequent trials remain in history until they pass comparison with it.
    latest=best if best.get('measured_improvement') else next((r for r in reversed(releases) if r.get('eligible_for_latest',True)),best)
    return {'latest':public(latest),'best':public(best),'baseline':public(original),
            'latest_selection':'best_verified_policy' if best.get('measured_improvement') else 'newest_reviewable_trial',
            'history':[public(r) for r in reversed(releases)],'issues':issues}
