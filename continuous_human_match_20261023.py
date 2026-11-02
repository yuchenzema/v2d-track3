"""Continuously audit all original videos and queue isolated corrective experiments."""
import fcntl
import hashlib
import json
import os
from pathlib import Path
import subprocess
import time
import traceback
import urllib.request
from improvement_common import BASE,BATCH,save,utc
from human_match_criteria import assess,LIMITS,CRITERIA_VERSION
from episode_completion_gate import completion,first_unfinished
import thorough_episode_recheck as human

OUT=Path(os.environ.get('V2D_QUEUE_DIR',BATCH/'human_match_loop_20261023'))
PY=BASE/'runs/episode_000005_newton_20261015/.venv-video/bin/python'
EXPECTED=[1,3,5,6,8,10,14,15,16,19,20,25,28,30,32,33,35,36,37,38]
PRIORITY=[19,20,30,33]
SERIAL_ORDER=list(dict.fromkeys([20,30,19,33]+EXPECTED))
FOCUS=[int(value) for value in os.environ['V2D_FOCUS_EPISODES'].split(',')] if os.environ.get('V2D_FOCUS_EPISODES') else EXPECTED
assert FOCUS and len(FOCUS)==len(set(FOCUS)) and set(FOCUS)<=set(EXPECTED),'Invalid focused episode scope'
if os.environ.get('V2D_FOCUS_EPISODES'):SERIAL_ORDER=FOCUS.copy()

def required_episode(state):
    return first_unfinished(SERIAL_ORDER,state.get('episodes',{})) if state.get('require_verified_match_before_next') else None

def round_attempt_counts(state):
    baseline=read(OUT/'recheck-request.json',{}).get('previous_attempt_counts',{})
    return {ep:max(0,sum(job['episode']==ep for job in state.get('optimization_history',[]))-baseline.get(str(ep),0)) for ep in EXPECTED}

def external_optimization():
    """Honor the independently queued episode20 without launching duplicates."""
    request=BATCH/'requested_episodes_20_30/episode20-after30-request.json'
    job=read(request,{})
    if not job:return None
    root=BASE/f"runs/episode_000020_newton_human_match_20261023_g{job['generation']:03d}"
    status=read(root/'research/rebuild-status.json',{})
    try:
        argv=Path(f"/proc/{int(status['pid'])}/cmdline").read_bytes()
        if b'execute_human_match_job_20261023.py' in argv and str(request).encode() in argv:
            return {'episode':20,'pid':status['pid'],'request':str(request),'run':str(root)}
    except (OSError,KeyError,ValueError):pass
    reservation=read(BATCH/'requested_episodes_20_30/status.json',{})
    try:
        argv=Path(f"/proc/{int(reservation['pid'])}/cmdline").read_bytes()
        if reservation.get('status')=='running_episode20' and b'sequential_30_then_20_20261026.py' in argv:
            return {'episode':20,'pid':reservation['pid'],'request':str(request),'run':str(root),'reservation_between_stages':True}
    except (OSError,KeyError,ValueError):pass
    return None

def next_generation(episode,minimum):
    generation=minimum
    while (BASE/f'runs/episode_{episode:06d}_newton_human_match_20261023_g{generation:03d}').exists() or (OUT/f'jobs/episode_{episode:06d}-g{generation:03d}.json').exists():
        generation+=1
    return generation

def import_external_result(state):
    request=BATCH/'requested_episodes_20_30/episode20-after30-request.json'
    outcome=read(request.with_suffix('.result.json'),{})
    if not outcome.get('all_pipeline_stages_executed'):return
    if any(job.get('request')==str(request) for job in state['optimization_history']):return
    job=read(request,{})
    job.update(request=str(request),exit_code=0,finished_utc=outcome.get('updated_utc'),
               finished_timestamp=time.time(),origin='external_episode20_priority_queue')
    if outcome.get('strict_check'):job['candidate_check']=compact(outcome['strict_check'])
    state['optimization_history'].append(job)

def read(path,default=None):
    try:return json.loads(path.read_text())
    except (OSError,ValueError):return default

def snapshot():
    with urllib.request.urlopen('http://127.0.0.1:8768/api/status',timeout=30) as response:return json.load(response)

def progress(completed,ep=None,frame=0,total=0,status='running',**extra):
    save(OUT/'audit-progress.json',{'updated_utc':utc(),'pid':os.getpid(),'status':status,'episode':ep,'frame':frame,'total_frames':total})

human.report_status=progress;human.__file__=__file__

def audit(row):
    ep=row['episode'];release=row['videos']['latest'];dst=OUT/f"checks/{release['version']}"
    path=dst/f'episode_{ep:06d}/strict-match.json'
    recroot=BASE/'runs/episode_000001_newton_v5_20261014' if ep==1 else Path(release['run'])
    rec=recroot/'reconstruction'
    watched=[BASE/f'datasets/video_to_data_challenge/track_3/evaluation/videos/chunk-000/observation.images.ego_cam_a/episode_{ep:06d}.mp4',
             rec/'scene_motion_physical.npz',rec/'hand_measurements.npz',rec/'camera_intrinsics.json',Path(release['recorded_states']),
             *sorted((rec/'masks').glob('*/*.png')),*sorted((rec/'mesh_priors').glob('*.ply')),
             BASE/'human_match_criteria.py',BASE/'thorough_episode_recheck.py',
             BASE/'episode_completion_gate.py',BASE/'rollout_collision_validation.py',
             BASE/'rigid_pose_collision_audit.py',BASE/'audit_recorded_collisions_native.py']
    if (OUT/'recheck-request.json').exists():watched.append(OUT/'recheck-request.json')
    def signature():return hashlib.sha256(json.dumps([(str(p),p.stat().st_size,p.stat().st_mtime_ns) for p in watched]).encode()).hexdigest()
    input_signature=signature()
    cached=read(path,{})
    if cached.get('criteria_version')==CRITERIA_VERSION and cached.get('thresholds')==LIMITS and cached.get('input_signature')==input_signature:return cached
    human.OUT=dst;dst.mkdir(parents=True,exist_ok=True)
    report=human.recheck({'episode':ep,'source_frames':row['source_frames'],'objects':row['objects'],'videos':{'latest':release}},[])
    data=read(dst/f'episode_{ep:06d}/frame-metrics.json')
    strict=assess(report,data)
    assert signature()==input_signature,'Audit inputs changed while scoring; retry against a stable input set'
    strict.update(updated_utc=utc(),run=release['run'],baseline_run=row['run'],strict_report='/artifacts/'+str(path.relative_to(BASE/'runs')),
                  original_video_sha256=report['source_video_sha256'],links=report['links'],input_signature=input_signature)
    save(path,strict);return strict

def compact(result):
    out=dict(result)
    out['objects']={name:{key:value for key,value in obj.items() if key not in ['per_frame_errors','failed_frames','unobservable_frames','reconstruction_mismatch_frames']} for name,obj in result['objects'].items()}
    return out

def initial_candidates(state):
    """Also check finished trials which were not selected as the current video."""
    state.setdefault('initial_candidate_checks',{})
    for ep in PRIORITY:
        folder=BATCH/'improvements_20261023_rebuild'
        result=read(folder/f'episode_{ep:06d}-fresh-result.json',{})
        report=read(folder/f'human_review/episode_{ep:06d}/report.json',{})
        frames=read(folder/f'human_review/episode_{ep:06d}/frame-metrics.json',{})
        if result.get('status')!='completed' or not report or not frames:continue
        previous=state['initial_candidate_checks'].get(str(ep),{})
        if previous.get('version')==report['version']:continue
        strict=assess(report,frames);strict.update(updated_utc=utc(),run=result['run'],links=report['links'])
        save(Path(result['run'])/'research/strict-human-match.json',strict)
        state['initial_candidate_checks'][str(ep)]=compact(strict)

def execution_plan(state,initial):
    """List every episode, including work waiting behind the priority batch."""
    plan={};priority_busy=initial.get('status') in ['running','retry_wait']
    priority_active=(initial.get('active') or {}).get('episode')
    corrective_active=(state.get('active_optimization') or {}).get('episode')
    for ep in EXPECTED:
        check=state.get('episodes',{}).get(str(ep),{})
        if ep not in FOCUS:status='parked_outside_focus'
        elif state.get('require_verified_match_before_next') and completion(check)['episode_done']:status='verified_episode_done'
        elif state.get('require_verified_match_before_next') and ep!=required_episode(state):status='waiting_for_current_episode_match'
        elif ep==(state.get('external_optimization') or {}).get('episode'):status='running_priority_pipeline'
        elif ep==corrective_active:status='running_corrective_cycle'
        elif state.get('dispatch_hold'):
            status='waiting_sam3d' if state['dispatch_hold'].get('method')=='SAM 3D Objects' else 'queued_corrective_cycle'
        elif priority_busy and ep in PRIORITY and str(ep) not in state.get('initial_candidate_checks',{}):
            status='running_priority_pipeline' if ep==priority_active else 'queued_priority_pipeline'
        elif check.get('passes_strict_local_checks'):status='monitoring_local_pass'
        elif check.get('recommended_action')=='visibility_review':status='visibility_unverified'
        else:status='queued_corrective_cycle'
        plan[str(ep)]={'episode':ep,'automatic':ep in FOCUS,'status':status,
                       'stages':['reconstruction','policy_training','rollout','original_video_recheck'],
                       'recheck_after_every_rollout':True,'exact_human_match_verified':False}
    return plan

def emit(state):
    state.update(updated_utc=utc(),pid=os.getpid(),process_script=Path(__file__).name,exact_human_match_verified=False)
    state['episodes_checked']=len(state.get('episodes',{}))
    checked=state.get('current_round_checked',[])
    checked=[ep for ep in checked if ep in FOCUS]
    state['current_round_episodes_checked']=len(checked)
    state['current_round_source_frames_checked']=sum(state['episodes'].get(str(ep),{}).get('source_frames',0) for ep in checked)
    state['strict_local_passes']=[int(ep) for ep,result in state.get('episodes',{}).items() if int(ep) in FOCUS and result['passes_strict_local_checks']]
    strict_advance=state.get('require_verified_match_before_next')
    state['episode_completion']={str(ep):completion(state.get('episodes',{}).get(str(ep),{})) for ep in EXPECTED}
    state['episodes_done']=[ep for ep in FOCUS if state['episode_completion'][str(ep)]['episode_done']]
    state['pending_episodes']=[ep for ep in FOCUS if not (state['episode_completion'][str(ep)]['episode_done'] if strict_advance else state.get('episodes',{}).get(str(ep),{}).get('passes_strict_local_checks'))]
    state['focused_episodes']=FOCUS
    state['parked_episodes']=[ep for ep in EXPECTED if ep not in FOCUS]
    initial=read(BATCH/'improvements_20261023_rebuild/automatic-queue.json',{})
    state['execution_plan']=execution_plan(state,initial)
    state['execution_mode']='one_episode_at_a_time'
    state['maximum_active_episodes']=1
    running=(state.get('external_optimization') or state.get('active_optimization') or {}).get('episode')
    order=([running] if running is not None else [])+[job['episode'] for job in state.get('deferred_optimizations',[])]
    counts=round_attempt_counts(state)
    order+=sorted(state['pending_episodes'],key=lambda ep:(counts[ep],SERIAL_ORDER.index(ep)))
    state['serial_queue_order']=list(dict.fromkeys(order))
    if strict_advance:
        state['required_episode']=required_episode(state)
        state['serial_queue_order']=[ep for ep in SERIAL_ORDER if ep in state['pending_episodes']]
    for ep,row in state['execution_plan'].items():
        row['queue_position']=state['serial_queue_order'].index(int(ep))+1 if int(ep) in state['serial_queue_order'] else None
    save(OUT/'execution-plan.json',{'updated_utc':utc(),'automatic':True,'episodes':state['execution_plan']})
    save(OUT/'status.json',state)

def public_validation_required(policy):
    # Only an explicit scope opt-out skips public validation. Evaluation
    # source, motion and contact gates remain independent and mandatory.
    return bool(policy.get('enabled')) and policy.get('public_validation_required') is not False


def main():
    # Own-geometry scope supersedes historical policy/scan replay requests.
    from own_geometry import required as own_required
    own=own_required(OUT)
    from fast_submission_policy import enabled as fast_submission_enabled
    if own and fast_submission_enabled(OUT):
        from fast_evaluation_queue import run as complete_first
        complete_first(OUT,FOCUS)
        return
    if own:
        from own_submission_queue import run as submit_own_cohort
        submit_own_cohort(OUT,FOCUS)
    if public_validation_required(read(OUT/'validated-pipeline-policy.json',{})):
        from public_validation_pipeline import worker,public_reconstruction_coverage_complete,evaluation_source_coverage_pending
        while public_validation_required(read(OUT/'validated-pipeline-policy.json',{})):
            if read(BATCH/'user-pause.json',{}).get('status')=='paused_by_user':return
            if worker(OUT,FOCUS):break
            policy=read(OUT/'validated-pipeline-policy.json',{})
            validation=read(OUT/'pipeline_validation_20261029/status.json',{})
            if (policy.get('evaluation_reconstruction_after_public_coverage') and
                    not read(OUT/'controller-reload-request.json',{}).get('pending') and
                    validation.get('status')=='public_reference_validation_quality_insufficient' and
                    evaluation_source_coverage_pending(OUT,FOCUS,policy) and
                    public_reconstruction_coverage_complete(validation.get('results',{}),policy.get('public_episodes',[]),
                                                           read(OUT/'frontend-revisions.json',{}))):
                from source_cad_queue import run as audit_evaluation_sources
                audit_evaluation_sources(OUT,FOCUS,coverage_only=True)
            time.sleep(60)
    from submission_replay_queue import run as replay_preserved_policies
    if not own and replay_preserved_policies(OUT,FOCUS):return
    if read(OUT/'experimental-training-policy.json',{}).get('enabled'):
        from evaluation_experiment_queue import run as run_experiments
        if run_experiments(OUT,FOCUS):return
    if read(OUT/'source-first-policy.json',{}).get('enabled'):
        from source_cad_queue import run
        if not run(OUT,FOCUS):return
    if own:
        # Never fall through into the historical scan-based policy dispatcher.
        return
    OUT.mkdir(exist_ok=True);state=read(OUT/'status.json',{})
    prior_active=state.get('active_optimization')
    state.update(status='checking_original_videos',active_optimization=None,expected_episodes=FOCUS,active=True,
                 phase=f'{len(FOCUS)} selected episodes: source audit and corrective reconstruction, policy training and rollout',
                 description='Source comparisons run continuously; failed motion checks remain unresolved.')
    state['dispatch_hold']=read(OUT/'dispatch-hold.json',{})
    policy=read(OUT/'advancement-policy.json',{})
    state['require_verified_match_before_next']=bool(policy.get('require_exact_motion_before_advancing'))
    state.setdefault('optimization_history',[]);state.setdefault('episodes',{})
    epoch=read(OUT/'recheck-request.json',{}).get('requested_utc')
    if state.get('audit_epoch')!=epoch:state.update(audit_epoch=epoch,current_round_checked=[])
    active=None
    if prior_active and (external_optimization() or (state['require_verified_match_before_next'] and prior_active['episode']!=required_episode(state))):
        deferred=state.setdefault('deferred_optimizations',[])
        if not any(job['request']==prior_active['request'] for job in deferred):deferred.insert(0,prior_active)
        prior_active=None
    if prior_active and Path(prior_active['request']).exists():
        with (OUT/'optimization.log').open('a') as log:
            active=subprocess.Popen([str(PY),'-u',str(BASE/'execute_human_match_job_20261023.py'),'--request',prior_active['request']],cwd=BASE,stdout=log,stderr=subprocess.STDOUT)
        prior_active['pid']=active.pid;state['active_optimization']=prior_active
    while True:
        if read(BATCH/'user-pause.json',{}).get('status')=='paused_by_user':
            state.update(status='paused_by_user',active=False);emit(state);return
        state['external_optimization']=external_optimization()
        import_external_result(state)
        target=required_episode(state)
        if state.get('require_verified_match_before_next') and target is None:
            state.update(status='all_episodes_verified',active=False,current_episode=None);emit(state);return
        if active is not None:
            if active.poll() is None:
                state.update(status='running_one_episode',current_episode=state['active_optimization']['episode'])
                save(OUT/'audit-progress.json',{'updated_utc':utc(),'status':'waiting_for_active_episode'})
                emit(state);time.sleep(15);continue
            job=state['active_optimization'];job.update(exit_code=active.returncode,finished_utc=utc(),finished_timestamp=time.time())
            outcome=read(Path(job['request']).with_suffix('.result.json'),{})
            if outcome.get('strict_check'):job['candidate_check']=compact(outcome['strict_check'])
            state['optimization_history'].append(job);state['active_optimization']=None;active=None
        if state['external_optimization']:
            state.update(status='waiting_for_episode20_to_finish',current_episode=20)
            save(OUT/'audit-progress.json',{'updated_utc':utc(),'status':'waiting_for_active_episode'})
            emit(state);time.sleep(15);continue
        deferred_index=next((i for i,job in enumerate(state.get('deferred_optimizations',[])) if not state.get('require_verified_match_before_next') or job['episode']==target),None)
        if deferred_index is not None:
            job=state['deferred_optimizations'].pop(deferred_index)
            with (OUT/'optimization.log').open('a') as log:
                active=subprocess.Popen([str(PY),'-u',str(BASE/'execute_human_match_job_20261023.py'),'--request',job['request']],cwd=BASE,stdout=log,stderr=subprocess.STDOUT)
            job['pid']=active.pid;state['active_optimization']=job;emit(state);continue
        rows=snapshot()['episodes'];assert [row['episode'] for row in rows]==EXPECTED
        for row in sorted(rows,key=lambda row:(row['episode'] not in PRIORITY,EXPECTED.index(row['episode']))):
            if row['episode'] not in FOCUS:continue
            if state.get('require_verified_match_before_next') and row['episode']!=target:continue
            try:
                state.update(status='checking_original_videos',current_episode=row['episode']);emit(state)
                result=audit(row)
                visual=read(OUT/f"visual-survey/episode_{row['episode']:06d}.json",{})
                if visual.get('version')==result['version'] and visual.get('manual_review_completed'):result['visual_review']=visual
                state['episodes'][str(row['episode'])]=compact(result)
                if row['episode'] not in state.setdefault('current_round_checked',[]):state['current_round_checked'].append(row['episode'])
                emit(state)
            except Exception:
                save(OUT/f"episode_{row['episode']:06d}-audit-error.json",{'updated_utc':utc(),'error':traceback.format_exc()})
        if active is not None and active.poll() is not None:
            job=state['active_optimization'];job.update(exit_code=active.returncode,finished_utc=utc(),finished_timestamp=time.time())
            outcome=read(Path(job['request']).with_suffix('.result.json'),{})
            if outcome.get('strict_check'):job['candidate_check']=compact(outcome['strict_check'])
            state['optimization_history'].append(job);state['active_optimization']=None;active=None
            # The next pass audits both the selected release and the new trial.
        initial=read(BATCH/'improvements_20261023_rebuild/automatic-queue.json',{})
        if not state.get('require_verified_match_before_next'):initial_candidates(state)
        initial_busy=initial.get('status') in ['running','retry_wait']
        state['dispatch_hold']=read(OUT/'dispatch-hold.json',{})
        pending=[r for r in state['episodes'].values() if r['episode'] in FOCUS and not r['passes_strict_local_checks']]
        if state.get('require_verified_match_before_next'):
            pending=[r for r in state['episodes'].values() if r['episode']==target and not completion(r)['episode_done']]
        # An independent priority worker may start or finish while the CPU
        # audits run. Refresh immediately before admitting another GPU job.
        state['external_optimization']=external_optimization()
        state.update(current_episode=None,episodes_checked=len(state['episodes']),
                     strict_local_passes=[int(ep) for ep,r in state['episodes'].items() if r['passes_strict_local_checks']],
                     pending_episodes=[r['episode'] for r in pending],initial_four_queue_active=initial_busy)
        if active is None and pending and not initial_busy and not state['dispatch_hold'] and not state['external_optimization']:
            free=os.statvfs(BASE);disk=free.f_bavail*free.f_frsize
            if disk<15_000_000_000:state['status']='waiting_for_disk_space'
            else:
                counts=round_attempt_counts(state)
                failed={(job['episode'],job.get('source_version')):job for job in state['optimization_history'] if job.get('exit_code')}
                eligible=[r for r in pending if r['recommended_action']!='visibility_review' and ((r['episode'],r['version']) not in failed or time.time()-failed[r['episode'],r['version']].get('finished_timestamp',0)>1800)]
                if eligible:
                    result=min(eligible,key=lambda r:(counts[r['episode']],SERIAL_ORDER.index(r['episode'])))
                    ep=result['episode'];generation=next_generation(ep,counts[ep]+1)
                    job={'episode':ep,'generation':generation,'action':result['recommended_action'],'source_version':result['version'],
                         'source_run':result['run'],'baseline_run':result['baseline_run'],'worst_windows':result['worst_windows'],
                         'started_utc':utc(),'finished_timestamp':time.time()}
                    validated=read(OUT/'validated-pipeline-policy.json',{})
                    if validated.get('enabled'):
                        validation=read(OUT/'pipeline_validation_20261029/status.json',{})
                        if public_validation_required(validated):
                            assert validation.get('evaluation_training_admitted'),'Public video reconstruction validation has not passed'
                        job.update(validated_video_pipeline=True,training_budget=12_800_000,time_advance_frames=0.,
                                   public_validation_manifest=str(OUT/'pipeline_validation_20261029/status.json'))
                    if result.get('visual_review'):
                        job['visual_review']=result['visual_review']
                        job['worst_windows']=[{'start':a,'end':b,'source':'manual human/policy comparison'} for a,b in result['visual_review']['focus_intervals']]+job['worst_windows']
                    prior=next((entry.get('candidate_check') for entry in reversed(state['optimization_history']) if entry['episode']==ep and entry.get('candidate_check')),state.get('initial_candidate_checks',{}).get(str(ep)))
                    if prior:
                        job['seed_reconstruction_run']=prior['run']
                        job['worst_windows']+=prior['worst_windows']
                        job['action']=prior['recommended_action']
                        job['feedback_from_previous_candidate']=True
                    job['require_full_sequence_pickup']=True
                    cad=read(OUT/'source-cad-status.json',{}).get('results',{}).get(str(ep),{})
                    if cad.get('local_overlap_failed_object_frames')==0 and cad.get('unobservable_object_frames')==0:
                        job['approved_source_cad_run']=cad['run']
                    job['worst_windows']=[{'start':0,'end':result['source_frames']-1,'source':'All-source-frame optimization requested by user'}]
                    if ep==20:
                        support=read(BATCH/'requested_episodes_20_30/episode20-object-support.json',{})
                        if support.get('physical_stability_verified'):
                            job['object_pixel_refinement']={'water_pitcher':support['refinement_settings']}
                    request=OUT/f'jobs/episode_{ep:06d}-g{generation:03d}.json';save(request,job)
                    with (OUT/'optimization.log').open('a') as log:
                        active=subprocess.Popen([str(PY),'-u',str(BASE/'execute_human_match_job_20261023.py'),'--request',str(request)],cwd=BASE,stdout=log,stderr=subprocess.STDOUT)
                    job['pid']=active.pid;job['request']=str(request);state['active_optimization']=job
        state['status']='optimizing_and_checking' if active else 'checking_while_episode20_runs' if state['external_optimization'] else 'waiting_for_sam3d' if state['dispatch_hold'] else 'waiting_for_initial_rebuilds' if initial_busy else 'blocked_by_current_episode_motion' if state.get('require_verified_match_before_next') else state.get('status','checking_original_videos')
        emit(state);time.sleep(30)

if __name__=='__main__':
    OUT.mkdir(exist_ok=True)
    with (OUT/'controller.lock').open('a') as lock:
        fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB);main()
