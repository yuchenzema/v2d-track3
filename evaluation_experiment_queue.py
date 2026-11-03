"""Dispatch one source-admitted pilot through the existing serial controller."""
import json
import subprocess
import time
from pathlib import Path
from improvement_common import BASE,BATCH,save,utc,digest
from experimental_training_admission import assess

PY=BASE/'runs/episode_000005_newton_20261015/.venv-video/bin/python'


def read(path):
    try:return json.loads(Path(path).read_text())
    except (OSError,ValueError):return {}


def execution_inputs():
    return {name:digest(BASE/name) for name in ['evaluation_experiment_queue.py','execute_human_match_job_20261023.py','own_geometry.py','preserved_release.py','source_instance_prompts.py','training_continuation.py','own_submission_queue.py','fast_submission_policy.py','fast_evaluation_queue.py','fast_policy_execution.py','fast_metric_source_pose.py','batch_worker.py',
        'rerun_four_episodes_20261023.py','install_source_cad_reference.py',
        'experimental_training_admission.py','source_cad_training_gate.py',
        'fit_full_trajectory_contacts_20261023.py','optimize_native_grasps.py','native_pinch_contacts.py',
        'apply_collision_transitions.py','rigid_pose_collision_audit.py',
        'approach_grasp_probe.py','grasp_physics_checks.py',
        'legacy_source_frame_playback.py','fresh_improvement_training.py','actor_transfer_training.py','grasp_training_admission.py',
        'batch_export.py','batch_generic_retarget.py','batch_coherent.py','evaluate_reference_controller.py','official_policy_suite.py',
        'export_reference_diagnostics.py','thorough_episode_recheck.py','audit_rebuilt_rollout_20261023.py',
        'own_source_quality.py','quality_contact_reference.py','source_policy_validation.py','own_policy_quality_training.py','quality_followups.py','quality_policy_overlay.py']}


def baseline_source(source):
    source=Path(source);seen=set()
    while str(source) not in seen:
        seen.add(str(source));cfg=read(source/'episode-config.json')
        if not cfg.get('experimental_only'):return source
        assert cfg.get('baseline_run'),'Experimental source lacks an original baseline'
        source=Path(cfg['baseline_run'])
    raise ValueError('Cyclic experimental baseline provenance')


def resumable_request(previous,source_attempt,status,episode,trajectory_sha256):
    """Resume an interrupted current job, never mutate a finished experiment."""
    saved=status.get('request')
    if not saved or status.get('status')!='running':return None
    finished_requests={entry.get('request') for entry in [previous,source_attempt,
        *previous.get('source_attempts',{}).values()] if 'exit_code' in entry}
    if saved in finished_requests:return None
    prior=read(saved)
    scientific=lambda hashes:{k:v for k,v in hashes.items() if k!='evaluation_experiment_queue.py'}
    if (prior.get('episode')==episode and prior.get('source_version')==trajectory_sha256
            and scientific(status.get('execution_input_hashes',{}))==scientific(execution_inputs())):
        return Path(saved)
    return None


def candidates(out,focus,policy):
    from own_geometry import required as own_required,validate as validate_own
    own=own_required(out)
    reports=read(out/'source-cad-status.json').get('results',{})
    registered={}
    for ep,path in read(out/'experimental-source-candidates.json').items():
        if int(ep) in focus:registered[ep]=read(Path(path))
    history=read(out/'experimental-training-status.json').get('attempts',{})
    result=[]
    for ep in focus:
        repaired=registered.get(str(ep),{})
        repaired_parent=(repaired.get('source_candidate_parent_trajectory_sha256')
            if repaired.get('reference_kind')=='source_revalidated_physical_candidate'
            and repaired.get('physical_projection_applied') is True and assess(repaired,policy)['admitted'] else None)
        seen=set()
        for report in [reports.get(str(ep),{}),registered.get(str(ep),{})]:
            if own:
                try:validate_own(Path(report['run']))
                except (KeyError,OSError,ValueError):continue
            signature=report.get('trajectory_sha256')
            if not signature or signature in seen:continue
            seen.add(signature)
            # Use the admitted, rechecked physical derivative of this exact
            # source. A newer source hash remains an independent candidate.
            if signature==repaired_parent:continue
            admission=assess(report,policy)
            if not admission['admitted']:continue
            latest=history.get(str(ep),{})
            previous=latest.get('source_attempts',{}).get(signature,{})
            if not previous and latest.get('trajectory_sha256')==signature:previous=latest
            # Scheduler bookkeeping cannot justify rerunning an unchanged,
            # already-finished scientific experiment or touching frozen inputs.
            scientific=lambda hashes:{k:v for k,v in hashes.items() if k!='evaluation_experiment_queue.py'}
            if previous and scientific(previous.get('execution_input_hashes',{}))==scientific(execution_inputs()):continue
            result.append((ep,report,admission))
    return sorted(result,key=lambda item:-item[2]['minimum_qualified_fraction'])


def run(out,focus):
    policy=read(out/'experimental-training-policy.json')
    if policy.get('enabled') is not True:return False
    if read(BATCH/'user-pause.json').get('status')=='paused_by_user':return True
    if read(out/'controller-reload-request.json').get('pending'):return True
    from continuous_human_match_20261023 import next_generation,external_optimization
    from source_cad_queue import source_for
    if external_optimization():return False
    available=candidates(out,focus,policy)
    if not available:
        blocked=read(out/'experimental-training-status.json')
        if blocked.get('status')!='pilot_blocked':return False
        from own_geometry import required as own_required
        if own_required(out) and not read(Path(blocked.get('run',''))/'episode-config.json').get('own_geometry_required'):
            return False
        # A failed pilot is an actionable blocker, not a reason to spend hours
        # silently repeating unrelated CAD fits. New execution inputs re-admit it.
        while read(out/'experimental-training-policy.json').get('enabled'):
            if read(BATCH/'user-pause.json').get('status')=='paused_by_user':return True
            if read(out/'controller-reload-request.json').get('pending'):return True
            if candidates(out,focus,read(out/'experimental-training-policy.json')):return run(out,focus)
            state=read(out/'status.json')
            detail=str(blocked.get('failure_summary') or blocked.get('latest_failure') or 'See pilot logs').strip().splitlines()[-1]
            completed=bool(blocked.get('run') and (Path(blocked['run'])/'flash_chord_coherent_actor_transfer/TRAINING_COMPLETED').exists())
            label='Experimental post-training validation/export blocked: ' if completed else 'Experimental training preparation blocked: '
            state.update(status='experimental_policy_blocked',current_episode=None,active_optimization=None,
                phase=label+detail,updated_utc=utc())
            save(out/'status.json',state);time.sleep(15)
        return False
    ep,report,admission=available[0]
    status=read(out/'experimental-training-status.json')
    status.setdefault('attempts',{})
    # Reuse an interrupted job (same source evidence), never spawn a duplicate.
    generation=next_generation(ep,1)
    source=baseline_source(source_for(ep))
    job=dict(episode=ep,generation=generation,source_run=str(source),baseline_run=str(source),
        action='experimental_source_admitted_pilot',source_version=report['trajectory_sha256'],
        approved_source_cad_run=report['run'],experimental_training=True,experimental_admission=admission,
        experimental_only=True,validated_video_pipeline=True,
        public_validation_manifest=str(out/'pipeline_validation_20261029/status.json'),
        training_budget=int(policy['training_budget']),time_advance_frames=0.,require_full_sequence_pickup=True,
        worst_windows=[dict(start=0,end=report['source_frames']-1,source='Full unwarped source sequence')])
    from own_geometry import required as own_required
    if own_required(out):
        own_policy=read(out/'own-geometry-policy.json')
        job['training_budget']=int(own_policy.get('training_budget',policy['training_budget']))
        job['own_geometry_required']=True
        job['own_training_continuation_allowed']=True
        job['own_training_maximum_steps']=int(own_policy.get('maximum_training_steps',12_800_000))
        job['own_training_round_increment']=int(own_policy.get('training_round_increment',5_760_000))
    job['producing_code_commit']=subprocess.check_output(['git','rev-parse','HEAD'],cwd=BASE,text=True).strip()
    request=out/f'jobs/episode_{ep:06d}-experimental-g{generation:03d}.json'
    latest=status.get('attempts',{}).get(str(ep),{})
    source_attempt=latest.get('source_attempts',{}).get(report['trajectory_sha256'],{})
    prior_request=resumable_request(latest,source_attempt,status,ep,report['trajectory_sha256'])
    if prior_request and prior_request.exists():
        prior=read(prior_request)
        if prior.get('episode')==ep and prior.get('source_version')==report['trajectory_sha256']:
            request=prior_request;job=prior;generation=job['generation']
    job.update(source_run=str(source),baseline_run=str(source))
    from actor_transfer_training import recipe as actor_recipe
    from own_geometry import required as own_required
    transfer=None if own_required(out) else actor_recipe(ep,BASE)
    if transfer:job['actor_transfer']=transfer
    job['execution_input_hashes']=execution_inputs()
    save(request,job)
    run_root=BASE/f'runs/episode_{ep:06d}_newton_human_match_20261023_g{generation:03d}'
    state=read(out/'status.json')
    state.update(status='experimental_policy_preparation',current_episode=ep,active=True,
        phase='Experimental CAD admission; physical reference and grasp checks before short policy training',
        active_optimization=dict(episode=ep,run=str(run_root),request=str(request),
            action='Experimental pilot: physical reference/grasp validation, short training and official rollout',
            dataset_split='evaluation',experimental_only=True))
    status.update(status='running',active_episode=ep,run=str(run_root),admission=admission,
                  request=str(request),updated_utc=utc(),strict_completion_count=0,
                  execution_input_hashes=execution_inputs())
    status.pop('latest_failure',None)
    status.pop('failure_summary',None)
    save(out/'status.json',state);save(out/'experimental-training-status.json',status)
    logpath=out/f'logs/experimental-episode-{ep:06d}-g{generation:03d}.log'
    logpath.parent.mkdir(exist_ok=True)
    with logpath.open('a') as log:
        child=subprocess.Popen([str(PY),'-u',str(BASE/'execute_human_match_job_20261023.py'),'--request',str(request)],
            cwd=BASE,stdout=log,stderr=subprocess.STDOUT)
        state['active_optimization']['pid']=child.pid
        while child.poll() is None:
            activity=read(run_root/'research/rebuild-status.json')
            improvement=read(run_root/'research/improvement-status.json')
            state['phase']=activity.get('phase',state['phase'])
            state['status']='experimental_policy_training' if improvement.get('status')=='training' else 'experimental_policy_preparation'
            if (run_root/'flash_chord_coherent_actor_transfer/TRAINING_COMPLETED').exists():
                state['status']='experimental_policy_validation'
            state.update(updated_utc=utc());save(out/'status.json',state);time.sleep(10)
    previous=status['attempts'].get(str(ep),{})
    source_attempts=dict(previous.get('source_attempts',{}))
    if previous.get('trajectory_sha256'):
        source_attempts.setdefault(previous['trajectory_sha256'],{k:v for k,v in previous.items() if k!='source_attempts'})
    attempt=dict(trajectory_sha256=report['trajectory_sha256'],request=str(request),
        run=str(run_root),exit_code=child.returncode,finished_utc=utc(),log=str(logpath),
        execution_input_hashes=status['execution_input_hashes'])
    prior=source_attempts.get(report['trajectory_sha256'],{})
    if prior:
        attempt['run_history']=[*prior.get('run_history',[]),{k:v for k,v in prior.items() if k!='run_history'}]
    source_attempts[report['trajectory_sha256']]=dict(attempt)
    status['attempts'][str(ep)]=dict(attempt,source_attempts=source_attempts)
    status.update(status='pilot_finished' if child.returncode==0 else 'pilot_blocked',active_episode=None,updated_utc=utc())
    if child.returncode:
        status['latest_failure']=read(request.with_suffix('.error.json')).get('error','See '+str(logpath))
        status['failure_summary']=str(status['latest_failure']).strip().splitlines()[-1]
        recent=sorted((run_root/'logs').glob('*.log'),key=lambda p:p.stat().st_mtime,reverse=True)
        for log in recent[:3]:
            with log.open('rb') as stream:
                stream.seek(max(0,log.stat().st_size-16384));lines=stream.read().decode(errors='replace').splitlines()
            errors=[line for line in lines if line.startswith(('ValueError:','AssertionError:','TypeError:','KeyError:','FileNotFoundError:','ModuleNotFoundError:'))]
            if errors:
                status['failure_summary']=errors[-1];status['failure_log']=str(log);break
        gate=read(run_root/'research/reference-nonpenetration-gate.json')
        if gate.get('passed') is False:
            status['failure_summary']=(f"Episode {ep} 接觸檢查失敗：最大穿透 {gate.get('maximum_penetration_mm')} mm；"
                f"門檻 {gate.get('numerical_epsilon_m',1e-6)*1000} mm。Learner 尚未啟動。")
        # A newly source-admitted reconstruction may still need bounded table
        # clearance. Revalidate a separate candidate against every source frame
        # rather than weakening exact reference binding or moving frozen inputs.
        if ('Physics retargeting moved objects away from source-aligned CAD' in str(status.get('latest_failure'))
                and policy.get('allow_source_revalidated_physical_candidate') is True):
            try:
                from episode_execution_guard import episode_slot
                from source_ray_clearance_candidate import build
                with episode_slot(run_root,ep):repaired,repair_admission=build(run_root)
                status['bounded_source_repair']=dict(candidate=str(repaired),admission=repair_admission)
                if repair_admission['admitted']:
                    status.update(status='source_revalidated_after_physics_clearance',failure_summary=None)
            except Exception as error:
                status['bounded_source_repair_error']=f'{type(error).__name__}: {error}'
    save(out/'experimental-training-status.json',status)
    state.update(active_optimization=None,current_episode=None,status=status['status'],updated_utc=utc())
    save(out/'status.json',state)
    return run(out,focus)
