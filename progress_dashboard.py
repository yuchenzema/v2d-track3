"""Read-only local dashboard for real Track 3 training jobs and artifacts."""
from __future__ import annotations
import argparse
import csv
import hashlib
from datetime import datetime, timezone
from http.server import SimpleHTTPRequestHandler, ThreadingHTTPServer
import json
import mimetypes
import os
from pathlib import Path
import re
import shlex
import subprocess
import threading
import time
from urllib.parse import quote, unquote, urlsplit
from batch_native import active_learners
from video_catalog import catalog
from public_reference_selection import latest_cad_attempt

BASE=Path(__file__).resolve().parent
RUNS=BASE/'runs'
BATCH=RUNS/'evaluation_batch_20261015'
IMPROVEMENTS=BATCH/'improvements_20261019'
JSON_CACHE={}
SNAPSHOT=None
SNAPSHOT_AT=0.
LOCK=threading.Lock()


def read_json(path,default=None):
    path=Path(path)
    try:
        mtime=path.stat().st_mtime_ns
        if path in JSON_CACHE and JSON_CACHE[path][0]==mtime:return JSON_CACHE[path][1]
        value=json.loads(path.read_text());JSON_CACHE[path]=(mtime,value);return value
    except (OSError,ValueError):return JSON_CACHE.get(path,(None,default))[1]


def timestamp(value):
    try:return datetime.fromisoformat(value.replace('Z','+00:00')).timestamp()
    except (AttributeError,ValueError):return None


def latest_metrics(path):
    try:
        with Path(path).open('rb') as stream:
            stream.seek(0,2);size=stream.tell();stream.seek(max(0,size-32768));lines=stream.read().decode(errors='replace').splitlines()
        for line in reversed(lines):
            try:
                value=json.loads(line)
                if 'additional_environment_steps' in value:return value
            except ValueError:pass
    except OSError:pass
    return {}


def process_map():
    pids=active_learners();containers={};jobs=[]
    try:
        command=['docker','ps','--no-trunc','--format','{{.ID}} {{.Names}}']
        result=subprocess.run(['sg','docker','-c',shlex.join(command)],capture_output=True,text=True,timeout=3,check=True)
        for line in result.stdout.splitlines():
            cid,name=line.split(maxsplit=1)
            match=re.fullmatch(r'v2d-([a-z0-9]+)-episode(\d+)-newton',name)
            if match:containers[cid]={'episode':int(match[2]),'kind':'improvement' if match[1].startswith(('rebuild','pickup')) else match[1] if match[1] in ['batch','improvement'] else 'grasp','container':name}
    except (OSError,ValueError,subprocess.SubprocessError):pass
    for pid in pids:
        try:group=Path(f'/proc/{pid}/cgroup').read_text()
        except OSError:continue
        found=next((data for cid,data in containers.items() if cid in group),None)
        if found:jobs.append({'pid':pid,**found})
    return jobs,pids


def artifact(path):
    return '/artifacts/'+quote(str(Path(path).relative_to(RUNS)),safe='/')


def frontend_version():
    return hashlib.sha256(b''.join((BASE/name).read_bytes() for name in
        ['progress_dashboard.html','latest_video_gallery.html','live_object_review.html','human_video_audit.html'])).hexdigest()[:16]


def review_activities():
    active=[]
    scheduling=read_json(BATCH/'improvements_20261022_rebuild/gpu-scheduling.json',{})
    for path in (BATCH/'improvements_20261021/activities').glob('*.json'):
        row=read_json(path,{})
        if row.get('status')!='running':continue
        if scheduling.get('reconstruction_scheduler_waiting_for_gpu') and row.get('pid')==scheduling.get('reconstruction_scheduler_pid'):continue
        if row.get('activity_kind') in ['complete_reconstruction_and_retraining','independent_policy_retraining'] and 'policy training' in row.get('phase','').lower():continue
        try:
            argv=Path(f'/proc/{int(row["pid"])}/cmdline').read_bytes().split(b'\0')
            expected_scripts={row['process_script']}
            if row.get('episode')==19 and row.get('activity_kind')=='complete_reconstruction_and_retraining':expected_scripts.add('resume_rebuilt_episode19_20261022.py')
            if any(Path(os.fsdecode(arg)).name in expected_scripts for arg in argv if arg):
                row=dict(row)
                if row.get('activity_kind')=='complete_reconstruction_and_retraining' and row.get('run'):
                    run=Path(row['run']);cfg=read_json(run/'episode-config.json',{})
                    preview=run/'videos/reconstruction-source-review.mp4'
                    if preview.is_file():row['reconstruction_preview']=artifact(preview)
                    if 'foundationpose' in row.get('phase',''):
                        counts=[f'{name.replace("_"," ")}: {len(list((run/"reconstruction/object_poses"/name).glob("*.json")))} / {cfg.get("frames",0)} source poses' for name in cfg.get('objects',[])]
                        row['progress_description']='; '.join(counts)+'; source fitting follows pose estimation'
                    elif 'segmentation' in row.get('phase',''):
                        row['progress_description']=f'{len(list((run/"reconstruction/masks/1").glob("*.png")))} masks; {len(list((run/"reconstruction/depth").glob("*.png")))} depth frames regenerated'
                if row.get('total_cases'):
                    done=int(row.get('completed_cases',0));total=int(row['total_cases'])
                    row.update(progress_percent=100.*done/total,
                               progress_description=f'{done} / {total} physical ablations completed; {row.get("phase", "checking next case")}')
                if row.get('progress_log'):
                    log=Path(row['progress_log'])
                    if log.is_relative_to(RUNS) and log.is_file():
                        with log.open('rb') as stream:
                            stream.seek(0,2);stream.seek(max(0,stream.tell()-65536));tail=stream.read().decode(errors='replace')
                        matches=re.findall(r'generation (\d+) stable (\d+)',tail)
                        if matches:
                            generation,stable=map(int,matches[-1]);total=int(row.get('search_generations',6))
                            row.update(progress_percent=min(100.,100.*(generation+1)/total),
                                progress_description=f'{generation+1} / {total} search generations checked; {stable} stable candidates in the last generation')
                        else:row['progress_description']='Preparing the physics model; no search generation finished yet'
                    else:row['progress_description']='Preparing or waiting for GPU admission'
                active.append(row)
        except (OSError,KeyError,ValueError):continue
    return active


def latest_review_results():
    latest={}
    scheduling=read_json(BATCH/'improvements_20261022_rebuild/gpu-scheduling.json',{})
    for path in (BATCH/'improvements_20261021/activities').glob('*.json'):
        row=read_json(path,{})
        if 'episode' not in row:continue
        ep=row['episode']
        if row.get('updated_utc','')>latest.get(ep,{}).get('updated_utc',''):
            latest[ep]={k:v for k,v in row.items() if k not in ['fit','result','results','error']}
            latest[ep]['report']=artifact(path)
            checks=[]
            for result in row.get('results',[]):
                if not isinstance(result,dict):continue
                checks.append({k:result[k] for k in ['eligible','mean_iou','completion_rate'] if k in result})
                if result.get('review') and Path(result['review']).is_file():
                    checks[-1]['report']=artifact(Path(result['review']))
            latest[ep]['candidate_checks']=checks
            if row.get('activity_kind')=='complete_reconstruction_and_retraining' and row.get('training_started') is False:
                latest[ep]['status']='queued_for_training'
            if row.get('activity_kind') in ['complete_reconstruction_and_retraining','independent_policy_retraining'] and row.get('phase')=='Independent fresh FlashSAC policy training' and not (Path(row['run'])/'research/training-admission.json').exists():
                latest[ep].update(status='queued_for_training',phase='Reconstruction complete; waiting for GPU capacity before fresh policy training')
            if scheduling.get('reconstruction_scheduler_waiting_for_gpu') and row.get('pid')==scheduling.get('reconstruction_scheduler_pid') and row.get('status')=='running':
                latest[ep].update(status='queued_for_gpu',scheduled_phase=row.get('phase'),phase='Reconstruction queued while episode19 trains and validates; automatic resume is enabled')
    return latest


def contact_phase_status():
    phase=read_json(BATCH/'improvements_20261020_contacts/phase-status.json',{})
    continuations={}
    for ep,name in [(16,'stationary-continuation-phase.json'),(10,'stable-cup-continuation-phase.json')]:
        continuation=read_json(BATCH/'improvements_20261020_contacts'/name,{})
        if continuation.get('phase')=='completed':
            continuations[str(ep)]=continuation
            phase={**phase,'failed':[item for item in phase.get('failed',[]) if item!=ep],
                   'running_or_queued':[item for item in phase.get('running_or_queued',[]) if item!=ep],
                   'completed':sorted(set(phase.get('completed',[])+[ep]))}
            if continuation.get('measured_improvement'):
                phase['measured_improvements']=sorted(set(phase.get('measured_improvements',[])+[ep]))
        elif continuation.get('phase') in ['waiting_for_saved_legacy_checkpoint','waiting_for_legacy_worker_handoff','waiting_for_saved_training_slot','starting','training']:
            continuations[str(ep)]=continuation
            phase={**phase,'phase':'contact_finetuning','failed':[item for item in phase.get('failed',[]) if item!=ep],
                   'running_or_queued':sorted(set(phase.get('running_or_queued',[])+[ep]))}
    if continuations:
        phase['reconstruction_continuations']=continuations
        phase['updated_utc']=max([phase.get('updated_utc','')]+[c.get('updated_utc','') for c in continuations.values()])
        if not phase.get('running_or_queued'):
            phase['phase']='completed_with_experiment_failures' if phase.get('failed') else 'completed'
    return phase


def hardware():
    result={}
    try:
        value=subprocess.run(['nvidia-smi','--query-gpu=name,memory.total,memory.used,utilization.gpu',
                              '--format=csv,noheader,nounits'],capture_output=True,text=True,timeout=3,check=True).stdout
        name,total,used,util=next(csv.reader(value.splitlines()))
        result['gpu']={'name':name.strip(),'total_gib':float(total)/1024,'used_gib':float(used)/1024,'utilization':float(util)}
    except (OSError,ValueError,StopIteration,subprocess.SubprocessError):result['gpu']=None
    memory={}
    for line in Path('/proc/meminfo').read_text().splitlines():
        name,value=line.split(':',1)
        if name in ['MemTotal','MemAvailable']:memory[name]=int(value.split()[0])*1024
    result['memory']={'total_gib':memory['MemTotal']/2**30,'used_gib':(memory['MemTotal']-memory['MemAvailable'])/2**30}
    disk=os.statvfs(BASE)
    result['storage']={'free_gb':disk.f_bavail*disk.f_frsize/1e9,'total_gb':disk.f_blocks*disk.f_frsize/1e9}
    return result


def training_row(root,row,jobs,kind,now):
    cfg=read_json(root/'episode-config.json',{})
    provenance=read_json(root/'coherent-training-provenance.json',{})
    outcome=read_json(root/'research/improvement-status.json',{})
    metrics=latest_metrics(root/'flash_chord_coherent_actor_transfer/metrics.jsonl')
    steps=int(metrics.get('additional_environment_steps',row.get('additional_training_steps',0)))
    done=(root/'flash_chord_coherent_actor_transfer/TRAINING_COMPLETED').exists()
    budget=int(provenance.get('additional_training_step_budget',cfg.get('contact_training_additional_budget',cfg.get('training_budget',65_001_472 if kind=='grasp' else 50_000_896 if kind=='improvement' else 100_003_840))))
    job=next((j for j in jobs if j['episode']==cfg.get('episode') and j['kind']==kind and
              (not cfg.get('native_container') or j['container']==cfg['native_container'])),None)
    metric_at=timestamp(metrics.get('utc'))
    pct=100. if done else min(100.,max(0.,100.*steps/max(1,budget)))
    state=read_json(root/'pipeline-status.json',{})
    failed=state.get('pipeline',{}).get('status')=='failed' and not row.get('rollout_export_completed')
    if kind in ['improvement','grasp']:failed=outcome.get('status')=='failed'
    status=('failed' if failed else 'complete' if row.get('rollout_export_completed') else
            'rendering' if done else 'training' if job else 'waiting')
    training_phase=metrics.get('training_phase',provenance.get('training_phase'))
    if kind=='grasp' and (root/'grasp_pretraining/TRAINING_COMPLETED').exists():training_phase='full'
    priority=read_json(BATCH/'training-priority.json',{})
    waiting_reason=('Grasp reconstruction review' if kind=='grasp' and not job and
                    cfg.get('episode') in priority.get('held_for_reconstruction_review',[]) else None)
    if not waiting_reason and kind=='grasp' and status=='waiting':waiting_reason='Waiting for GPU capacity'
    if cfg.get('training_curriculum')=='fresh_rebuild' and not job and not done:
        rebuilding=read_json(root/'research/rebuild-status.json',{})
        waiting_reason=rebuilding.get('phase','Queued for complete reconstruction')
        if waiting_reason=='Independent fresh FlashSAC policy training':
            waiting_reason='Waiting for GPU capacity before fresh policy training'
        scheduling=read_json(BATCH/'improvements_20261022_rebuild/gpu-scheduling.json',{})
        if rebuilding.get('status')=='running' and scheduling.get('reconstruction_scheduler_waiting_for_gpu') and rebuilding.get('pid')==scheduling.get('reconstruction_scheduler_pid'):
            waiting_reason='Queued while episode19 trains and validates; automatic GPU scheduling'
    if kind=='grasp' and not job and not done and (root/'research/paused-for-episodes10-16.json').exists() and priority.get('deferred_for_episode10_16_compute'):
        status='waiting';waiting_reason='GPU priority for episodes10 and16; saved learner checkpoint retained'
    result={'episode':cfg.get('episode',row.get('episode')),'kind':kind,'objects':cfg.get('objects',[]),
            'steps':steps,'budget':budget,'percent':pct,'training_complete':done,
            'status':status,'pid':job['pid'] if job else None,'active':job is not None,
            'last_metric_utc':metrics.get('utc'),'heartbeat_age_seconds':max(0,now-metric_at) if metric_at else None,
            'training_phase':training_phase,'waiting_reason':waiting_reason,
            'backward_stage':metrics.get('backward_stage',provenance.get('backward_stage')),
            'training_start_source_frame':metrics.get('training_start_source_frame',provenance.get('training_start_source_frame')),
            'adaptive_training_start_source_frame':metrics.get('curriculum/adaptive_reset_min_source_frame'),
            'steps_per_second':metrics.get('throughput/env_steps_per_second'),
            'source_frames':cfg.get('frames',0),'quality':row.get('quality_assessment'),
            'export_complete':bool(row.get('rollout_export_completed')),
            'run':str(root),'links':{}}
    reviews=read_json(root/'checkpoint-reviews.json',[])
    result.update(updated_utc=outcome.get('updated_utc',metrics.get('utc')),
                  measured_improvement=outcome.get('measured_improvement'),
                  latest_checkpoint_review=({k:reviews[-1].get(k) for k in
                      ['environment_steps','completion_rate_32_worlds','source_mean_object_iou']} if reviews else None))
    if (root/'videos/reconstruction-source-review.mp4').is_file():
        result['reconstruction_preview']=artifact(root/'videos/reconstruction-source-review.mp4')
    reconstruction_record=root/'research/full-reconstruction-completed.json'
    reused=cfg.get('reconstruction_reused_from')
    if reused:
        reconstruction_record=Path(reused)/'research/full-reconstruction-completed.json'
    if reconstruction_record.is_file():
        fidelity=read_json(root/'evaluation/reference-source-fidelity.json',{})
        result['reconstruction_check']={
            'completed':True,'reused_for_additional_policy_training':bool(reused),
            'source_frames_scored':len(fidelity.get('frames',[])),
            'source_frames_expected':cfg.get('frames',0),
            'report':artifact(reconstruction_record),
            'frame_report':artifact(root/'evaluation/reference-source-fidelity.json')}
        contact_fit=root/'research/full-trajectory-contact-fit.json'
        fit=read_json(contact_fit,{})
        if fit.get('grasp_stability_issues_remain'):
            result['reconstruction_check']['grasp_warning']='Unassisted static grasp tests still fail. These are reference diagnostics; the learned policy must be checked separately.'
            result['reconstruction_check']['grasp_report']=artifact(contact_fit)
    visual_diagnosis=root/'research/checkpoint-diagnosis/pickup-visual-review.json'
    if visual_diagnosis.is_file():
        review=read_json(visual_diagnosis,{})
        figure=root/review.get('figure','')
        if figure.is_file():
            result['pickup_visual_review']={'finding':review.get('finding'),
                                          'figure':artifact(figure),'report':artifact(visual_diagnosis)}
    diagnosis_path=root/'research/checkpoint-diagnosis/latest.json'
    diagnosis=read_json(diagnosis_path,{})
    if diagnosis and Path(diagnosis.get('figure','')).is_file():
        result['checkpoint_diagnosis']={'steps':diagnosis['checkpoint_steps'],'report':artifact(diagnosis_path),'figure':artifact(Path(diagnosis['figure']))}
    if result['export_complete']:
        result['links']={'comparison':artifact(root/'videos/original_vs_policy.mp4'),
                         'rollout':artifact(root/'videos/policy_rollout_source_timed.mp4'),
                         'report':artifact(root/'result-report.json')}
    return result


def collect():
    global SNAPSHOT,SNAPSHOT_AT
    with LOCK:
        now=time.time()
        if SNAPSHOT is not None and now-SNAPSHOT_AT<4 and SNAPSHOT.get('frontend_version')==frontend_version():return SNAPSHOT
        batch=read_json(BATCH/'batch-status.json',{})
        audit=read_json(BATCH/'batch-final-audit.json',{})
        phase=read_json(IMPROVEMENTS/'phase-status.json',{'phase':'not_started'})
        priorities=read_json(IMPROVEMENTS/'user-priorities.json',{}).get('requested_episodes',[])
        staging=read_json(BATCH/'submission_staging/readiness.json',{})
        precision=read_json(BATCH/'improvements_20261021/phase-status.json',{})
        current_runs=read_json(BATCH/'improvements_20261021/active-runs.json',{})
        active_reviews=review_activities()
        latest_reviews=latest_review_results()
        human_recheck=read_json(BATCH/'human_video_recheck_20261021/status.json',{})
        human_recheck={k:v for k,v in human_recheck.items() if k!='episodes'}
        if human_recheck:
            human_recheck['report']=artifact(BATCH/'human_video_recheck_20261021/status.json')
            try:human_recheck['active']=human_recheck.get('status')=='running' and b'thorough_episode_recheck.py' in Path(f'/proc/{human_recheck.get("pid",0)}/cmdline').read_bytes()
            except OSError:human_recheck['active']=False
        jobs,pids=process_map()
        episodes=[training_row(Path(row['run']),row,jobs,'batch',now) for row in batch.get('episodes',[])]
        for ep in (1,5):
            legacy=read_json(BATCH/'all20_evaluation'/f'episode_{ep:06d}-release.json')
            if legacy:
                episodes.append({'episode':ep,'kind':'standalone','objects':legacy['objects'],
                    'steps':legacy['environment_steps'],'budget':legacy['environment_steps'],'percent':100.,
                    'training_complete':True,'status':'complete','pid':None,'active':False,
                    'last_metric_utc':None,'heartbeat_age_seconds':None,'training_phase':None,'waiting_reason':None,
                    'backward_stage':None,'training_start_source_frame':None,'adaptive_training_start_source_frame':None,
                    'steps_per_second':None,'source_frames':legacy['source_frames'],'quality':legacy['quality'],
                    'export_complete':True,'run':legacy['run'],'links':{},'artifact_verified':True})
        episodes.sort(key=lambda row:row['episode'])
        active=[]
        for row in episodes:
            if row['kind']!='standalone':row['artifact_verified']=audit.get('episodes',{}).get(f'{row["episode"]:06d}',{}).get('integrity_passed',False)
            if row['active']:active.append(row)
        fresh=[]
        experiment_configs=[p for p in RUNS.glob('episode_*_newton_*/episode-config.json')
                            if not p.parent.name.endswith('_newton_batch_20261015')]
        for path in sorted(experiment_configs):
            root=path.parent
            if (root/'research/superseded-training.json').exists():continue
            if not (root/'research/improvement-status.json').exists():continue
            exported=(root/'deliverables/manifest.json').exists() and (root/'result-report.json').exists()
            report=read_json(root/'result-report.json',{}) if exported else {}
            cfg=read_json(path,{})
            kind='improvement' if cfg.get('native_container','').startswith('v2d-improvement-') or cfg.get('training_curriculum')=='fresh_rebuild' else 'grasp'
            item=training_row(root,{'rollout_export_completed':exported,'quality_assessment':report.get('quality_assessment')},jobs,kind,now)
            fresh.append(item)
            if item['active']:active.append(item)
        controller_results=[read_json(path,{}) for path in IMPROVEMENTS.glob('episode_*-controller-result.json')]
        fresh_results=[read_json(path,{}) for path in IMPROVEMENTS.glob('episode_*-fresh-result.json')]
        extra_phase=read_json(BATCH/'improvements_20261020/phase-status.json',{})
        extra_results=[read_json(path,{}) for path in (BATCH/'improvements_20261020').glob('episode_*-combination-result.json')]
        grasp_phase=read_json(BATCH/'improvements_20261020_grasp/phase-status.json',{})
        grasp_phase={**grasp_phase,'held_for_reconstruction_review':read_json(BATCH/'training-priority.json',{}).get('held_for_reconstruction_review',[])}
        grasp_results=[read_json(path,{}) for path in (BATCH/'improvements_20261020_grasp').glob('episode_*-fresh-result.json')]
        grasp_results += [read_json(path,{}) for path in (BATCH/'improvements_20261020_backward').glob('episode_*-fresh-result.json')]
        issues=[]
        for row in episodes:
            candidates=[r for r in fresh if r['episode']==row['episode']]
            current_name=current_runs.get(str(row['episode']))
            trial=(next((r for r in candidates if r['active']),None) or
                   next((r for r in candidates if Path(r['run']).name==current_name),None) or
                   max(candidates,key=lambda r:timestamp(r.get('updated_utc')) or 0,default=None))
            row['latest_trial']=trial
            row['optimization_status']=trial['status'] if trial else 'queued' if row['episode'] in priorities else None
            row['optimization_waiting_reason']=trial.get('waiting_reason') if trial else None
            row['optimization_percent']=trial['percent'] if trial else None
            row['optimization_active']=bool(trial and trial['active'])
            review_root=RUNS/f'episode_{row["episode"]:06d}_newton_grasp_curriculum_20261020'
            review=read_json(review_root/'research/grasp-refinement-progress.json',{})
            review_proc=Path(f'/proc/{review.get("pid",0)}/cmdline')
            try:review_active=review.get('status')=='checking_approach' and b'grasp_approach_search.py' in review_proc.read_bytes()
            except OSError:review_active=False
            row['reconstruction_review']=dict(review,active=bool(review_active)) if review else None
            if review_active and not row['optimization_active']:
                row.update(optimization_status='reconstructing',optimization_waiting_reason=review['description'],
                           optimization_active=True,optimization_percent=None)
            videos=catalog(row['episode'],row['run'])
            row['videos']=videos
            recheck=read_json(BATCH/f'human_video_recheck_20261021/episode_{row["episode"]:06d}/report.json',{})
            new_checks=[read_json(path,{}) for path in BATCH.glob(f'improvements_*/human_review/episode_{row["episode"]:06d}/report.json')]
            new_recheck=max(new_checks,key=lambda r:r.get('updated_utc','')) if new_checks else {}
            row['fresh_training_human_review']=new_recheck or None
            current_version=(videos.get('latest') or {}).get('version')
            if current_version:
                check_root=BATCH/f'human_match_loop_20261023/checks/{current_version}/episode_{row["episode"]:06d}'
                current_check=read_json(check_root/'report.json',{})
                visual_path=BATCH/f'human_match_loop_20261023/visual-survey/episode_{row["episode"]:06d}.json'
                visual=read_json(visual_path,{})
                if current_check and visual.get('version')==current_version and visual.get('manual_review_completed'):
                    visual=dict(visual,correction_focus='Original source frame intervals: '+', '.join(f'{a}–{b}' for a,b in visual['focus_intervals']))
                    current_check['visual_review']=visual
                    current_check['links']['survey']=artifact(visual_path.with_suffix('.jpg'))
                if current_check:new_checks.append(current_check)
                row['human_match_check']=read_json(check_root/'strict-match.json',{}) or None
            checked=[r for r in [recheck,*new_checks] if r.get('version')==(videos.get('latest') or {}).get('version')]
            row['human_video_recheck']=max(checked,key=lambda r:r.get('updated_utc','')) if checked else None
            if videos['latest']:
                latest=videos['latest']
                row['baseline_quality']=row['quality']
                row['quality']=latest['quality']
                row['links']=latest['links']
                row['video_label']=latest['label']
                row['video_version']=latest['version']
                row['artifact_verified']=latest['integrity_verified']
            if trial:
                accepted=videos.get('best') or {}
                row['optimization_outcome']=('accepted_improvement' if trial['status']=='complete' and accepted.get('run')==trial['run'] and accepted.get('measured_improvement') else
                    'completed_without_new_accepted_improvement' if trial['status']=='complete' else trial['status'])
                row['latest_training_trial_outcome']=row['optimization_outcome']
                latest=videos.get('latest') or {}
                if latest.get('measured_improvement') and (timestamp(latest.get('validated_utc')) or 0)>(timestamp(trial.get('updated_utc')) or 0):
                    row['optimization_outcome']='accepted_improvement'
                if trial['status']=='failed':issues.append(f'Episode {row["episode"]:06d}: current optimization trial failed; previous video retained.')
            ongoing=next((r for r in active_reviews if r['episode']==row['episode']),None)
            row['latest_review']=latest_reviews.get(row['episode'])
            if ongoing and not (trial and trial['active'] and ongoing.get('activity_kind')=='complete_reconstruction_and_retraining'):
                row.update(optimization_status='evaluating_controller',optimization_outcome='evaluating_controller',
                    optimization_active=True,optimization_percent=ongoing.get('progress_percent'),
                    optimization_waiting_reason=ongoing.get('progress_description') or ongoing.get('phase') or ongoing['description'])
            elif row['latest_review']:
                last=row['latest_review']
                if last.get('status')=='failed':issues.append(f'Episode {row["episode"]:06d}: latest physical diagnostic failed; previous video retained. {last.get("phase","")}')
                elif any(last.get(k) is False for k in ['stable_hold_and_lift','stable_grasp_found']):
                    row['optimization_outcome']='needs_reconstruction'
                elif last.get('eligible_for_independent_publication') is False:
                    row['optimization_outcome']='completed_without_new_accepted_improvement'
            issues.extend(f'Episode {row["episode"]:06d}: {issue}' for issue in videos['issues'])
        for row in episodes+fresh:
            if row in episodes and row['status']=='failed':issues.append(f'Episode {row["episode"]:06d}: baseline worker reported a failure.')
            if row['active'] and row['steps']>0 and row['heartbeat_age_seconds'] is not None and row['heartbeat_age_seconds']>180:
                issues.append(f'Episode {row["episode"]:06d}: training process exists, but its metrics have not updated for {int(row["heartbeat_age_seconds"])} seconds.')
        if human_recheck:
            human_recheck['current_versions_checked']=sum(bool(r.get('human_video_recheck')) for r in episodes)
            human_recheck['outdated_or_pending_episodes']=[r['episode'] for r in episodes if not r.get('human_video_recheck')]
            human_recheck['all_current_release_versions_rechecked']=human_recheck['current_versions_checked']==20
        if phase.get('phase')=='failed':issues.append('Improvement queue reported an error; see its detailed status.')
        if precision.get('failed'):issues.append('Current precision round needs attention: '+', '.join(map(str,precision['failed'])))
        attention=read_json(IMPROVEMENTS/'supervisor-attention.json')
        if attention:issues.append('Improvement recovery needs attention: '+attention.get('reason','check supervisor status'))
        queue_path=BATCH/'improvements_20261023_rebuild/automatic-queue.json'
        automatic_queue=read_json(queue_path,{})
        focus_scope=read_json(BATCH/'focus-scope.json',{})
        queue_dir=Path(focus_scope['queue_dir']) if focus_scope.get('status')=='active' and focus_scope.get('queue_dir') else BATCH/'human_match_loop_20261023'
        if not queue_dir.resolve().is_relative_to(BATCH.resolve()):queue_dir=BATCH/'human_match_loop_20261023'
        human_match_loop=read_json(queue_dir/'status.json',{})
        source_cad=read_json(queue_dir/'source-cad-status.json',{})
        public_validation=read_json(queue_dir/'pipeline_validation_20261029/status.json',{})
        if public_validation:
            public_validation['report']=artifact(queue_dir/'pipeline_validation_20261029/status.json')
            failure=Path(public_validation.get('error_report') or '/nonexistent')
            if failure.is_file():
                failure_text=failure.read_text(errors='replace');messages=failure_text.splitlines()
                nested=failure.parent/'generic-scene-refinement.log'
                if nested.is_file():messages+=nested.read_text(errors='replace').splitlines()
                cause=next((line for line in reversed(messages) if line.startswith(('RuntimeError:','AssertionError:','ValueError:'))),messages[-1] if messages else 'See failure log')
                plane_log=failure.parent/'initial-table-plane.log'
                if 'public-generic failed' in cause and plane_log.is_file() and 'assert best.sum()>100' in plane_log.read_text(errors='replace'):
                    cause='Table-plane fit failed: more than 100 source inliers required'
                match=re.search(r'public_episode_(\d+)',str(failure))
                stage=re.search(r'(public-[a-z-]+) failed',failure_text)
                markers={'public-generic':'scene_motion_physical.npz','public-metric':'stereo-depth-alignment.json',
                         'public-masks':'masks-reviewed.json','public-stereo':'foundation-stereo-sequence.json',
                         'public-odometry':'sift-pnp-odometry.json'}
                marker=(failure.parent.parent/'reconstruction'/markers[stage.group(1)] if stage and stage.group(1) in markers else None)
                resolved=bool(marker and marker.is_file() and marker.stat().st_mtime>failure.stat().st_mtime)
                if resolved:cause='Previous stage failure; the stage retried successfully'
                public_validation['last_failed_attempt']=dict(episode=int(match.group(1)) if match else None,reason=cause,resolved=resolved,report=artifact(failure))
            for result in public_validation.get('results',{}).values():
                latest=latest_cad_attempt(result)
                result['latest_cad_attempt']={k:latest.get(k) for k in ['refined_run','refined_metrics','repair_round','selection_reason']}
                result['latest_cad_attempt']['selected_as_reference']=latest is result and result.get('adopted_over_baseline',False)
                overlay=Path(latest['refined_run'])/'videos/cad-source-overlay.mp4'
                if overlay.is_file():result['overlay_link']=artifact(overlay)
                selected=result.get('selected_reference_prediction')
                retained_root=(Path(selected).parent.parent if selected else Path(result['refined_run']) if result.get('adopted_over_baseline') else None)
                retained_overlay=retained_root/'videos/cad-source-overlay.mp4' if retained_root else None
                if retained_overlay and retained_overlay.is_file():result['retained_overlay_link']=artifact(retained_overlay)
            if public_validation.get('run'):
                public_root=Path(public_validation['run'])
                public_cfg=read_json(public_root/'episode-config.json',{})
                public_cad=Path(public_validation.get('cad_run') or public_root.with_name(public_root.name.replace('_pipeline_','_cad_alignment_')))
                if public_validation.get('active_episode') is not None and not public_cad.name.startswith(f"public_episode_{int(public_validation['active_episode']):06d}_"):
                    public_cad=public_root.with_name(public_root.name.replace('_pipeline_','_cad_alignment_')+'_v2')
                public_validation['cad_progress']=read_json(public_cad/'research/cad-progress.json',{})
                if public_validation.get('phase')=='public-cad-refinement' and public_validation['cad_progress'].get('phase'):
                    public_validation['phase_detail']=public_validation['cad_progress']['phase']
                elif public_validation.get('phase')=='public-metric' and public_cfg.get('frames'):
                    total=public_cfg['frames'];rec=public_root/'reconstruction'
                    propagation=read_json(rec/'masks/propagation-report.json',{})
                    tracks=propagation.get('objects',{})
                    masks_done=(len(tracks)==len(public_cfg.get('objects',[])) and all(row.get('all_source_frames_propagated') and row.get('frames')==total for row in tracks.values()))
                    depth_count=sum(1 for _ in (rec/'depth_stereo').glob('[0-9][0-9][0-9][0-9][0-9][0-9].png'))
                    public_validation['phase_detail']=(f'SAM2 propagated all {total} source frames per object; metric stereo depth {depth_count} / {total} frames' if masks_done else 'Propagating source object masks before metric depth')
                elif public_validation.get('phase')=='public-generic' and public_cfg.get('frames'):
                    counts={name:sum(1 for _ in (public_root/'reconstruction/object_poses'/name).glob('[0-9][0-9][0-9][0-9][0-9][0-9].json')) for name in public_cfg.get('objects',[])}
                    pending=next((name for name,count in counts.items() if count<public_cfg['frames']),None)
                    public_validation['phase_detail']=(f'Object poses {pending}: {counts[pending]} / {public_cfg["frames"]} frames' if pending else 'Combining measured camera, object poses and hand tracks')
                else:public_validation.pop('phase_detail',None)
                public_validation.pop('preview',None)
                public_validation.pop('reference_baseline',None)
                revision=read_json(queue_dir/'validated-pipeline-policy.json',{}).get('score_revision','')
                revision+=public_cfg.get('frontend_revision','')
                baseline=queue_dir/f"pipeline_validation_20261029/public_{int(public_validation.get('active_episode') or 0):06d}{revision}_reference_baseline.json"
                if baseline.exists():public_validation['reference_baseline']=read_json(baseline,{})
                if (public_cad/'research/cad-live-preview.jpg').exists():public_validation['preview']=artifact(public_cad/'research/cad-live-preview.jpg')
                for name in ['research/cad-live-preview.jpg','reconstruction/frames/000000.png']:
                    if not public_validation.get('preview') and (public_root/name).exists():public_validation['preview']=artifact(public_root/name);break
        if source_cad:
            source_cad['report']=artifact(queue_dir/'source-cad-status.json')
            for cad_result in source_cad.get('results',{}).values():
                cad_result['overlay_link']=artifact(Path(cad_result['overlay']))
                cad_result['frames_link']=artifact(Path(cad_result['frames_report']))
            if source_cad.get('run'):
                cad_root=Path(source_cad['run'])
                source_cad['progress']=read_json(cad_root/'research/cad-progress.json',source_cad.get('progress',{}))
                if (cad_root/'research/cad-live-preview.jpg').exists():source_cad['preview']=artifact(cad_root/'research/cad-live-preview.jpg')
        session_updates=read_json(queue_dir/'session-updates.json',{})
        dataset_resource_audit=read_json(queue_dir/'huggingface_audit_20261028/summary.json',{})
        if dataset_resource_audit:dataset_resource_audit['report']=artifact(queue_dir/'huggingface_audit_20261028/resource-report.json')
        locating_spheres={}
        sphere_jobs=[human_match_loop.get('active_optimization') or {}]+list(reversed(human_match_loop.get('optimization_history',[])))[0:8]
        for job in sphere_jobs:
            if not job.get('episode') or job.get('generation') is None:continue
            sphere_root=RUNS/f"episode_{job['episode']:06d}_newton_human_match_20261023_g{job['generation']:03d}"
            locating_spheres=read_json(sphere_root/'research/locating-sphere-tracking.json',{})
            if locating_spheres:
                fitting=read_json(sphere_root/'research/source-pixel-pose-refinement-v2-spheres.json',{})
                locating_spheres['used_in_pose_fitting']=bool(fitting.get('locating_spheres_used_as_image_references'))
                locating_spheres['report']=artifact(sphere_root/'research/locating-sphere-tracking.json')
                break
        if source_cad.get('enabled'):locating_spheres={}
        collision_policy=read_json(queue_dir/'collision-policy.json',{})
        for row in episodes:
            if row['episode']==32 and collision_policy.get('episode32_known_reference_issue'):
                row['known_reference_penetration_mm']=collision_policy['episode32_known_reference_issue']['maximum_penetration_mm']
        queue_health=read_json(queue_dir/'health-monitor.json',{})
        collision_validation={}
        if queue_health.get('run'):
            collision_root=Path(queue_health['run'])
            if collision_root.resolve().is_relative_to(RUNS.resolve()):
                collision_validation=read_json(collision_root/'research/rollout-collision-validation.json',{})
        if queue_health:
            queue_health['report']=artifact(queue_dir/'health-monitor.json')
            try:health_age=max(0,time.time()-datetime.fromisoformat(queue_health['updated_utc'].replace('Z','+00:00')).timestamp())
            except (KeyError,TypeError,ValueError):health_age=None
            queue_health['heartbeat_age_seconds']=health_age
            if health_age is not None and health_age>180:
                issues.append('Independent queue health monitor has not reported for more than 3 minutes.')
            issues.extend('Queue monitor: '+message for message in queue_health.get('alerts',[]))
        sam3d=read_json(BATCH/'sam3d_rebuild_20261026/status.json',{})
        pickup_repair=read_json(BATCH/'episode30_pickup_repair_20261026/status.json',{})
        pose_root=RUNS/'episode_000020_newton_human_match_20261023_g004'
        pose_records=read_json(pose_root/'research/triangle-refit-records.json',[])
        source_pose_repair={}
        if pose_records:
            source_pose_repair={'episode':20,'frames_processed':len(pose_records),'total_frames':141,
                'mean_iou_before':sum(r['source_iou_before'] for r in pose_records)/len(pose_records),
                'mean_iou_after':sum(r['source_iou_after'] for r in pose_records)/len(pose_records),
                'completed':(pose_root/'research/triangle-pitcher-source-fit.json').exists(),
                'details':artifact(pose_root/'research/triangle-refit-records.json')}
            preview=pose_root/'research/triangle-source-fit-review.jpg'
            if preview.exists():source_pose_repair['preview']=artifact(preview)
            temporal=read_json(pose_root/'research/temporal-refit-records.json',[])
            if temporal:
                source_pose_repair.update(temporal_frames_processed=len(temporal),
                    temporal_mean_iou=sum(row['source_iou'] for row in temporal)/len(temporal),
                    temporal_max_step_degrees=max(row['angular_step_rad'] for row in temporal)*180/3.141592653589793,
                    temporal_completed=(pose_root/'research/temporal-pitcher-source-fit.json').exists())
                preview=pose_root/'research/temporal-source-fit-review.jpg'
                if preview.exists():source_pose_repair['preview']=artifact(preview)
            holds=read_json(pose_root/'research/physical-holds-verified.json',{})
            source_pose_repair['physical_holds_verified']=holds.get('both_objects_static_hold_verified') is True
            source_pose_repair['policy_training_started']=(pose_root/'coherent-input-checksums.json').exists()
            source_pose_repair['policy_training_completed']=(pose_root/'flash_chord_coherent_actor_transfer/TRAINING_COMPLETED').exists()
        requested_work=read_json(BATCH/'requested_episodes_20_30/status.json',{})
        if requested_work:
            requested_work['report']=artifact(BATCH/'requested_episodes_20_30/status.json')
            if requested_work.get('status')=='running_episode20':
                request=read_json(BATCH/'requested_episodes_20_30/episode20-after30-request.json',{})
                generation=request.get('generation')
                if generation is not None:
                    work_root=RUNS/f'episode_000020_newton_human_match_20261023_g{generation:03d}'
                    rebuilt=(work_root/'research/full-reconstruction-completed.json').exists()
                    trained=(work_root/'flash_chord_coherent_actor_transfer/TRAINING_COMPLETED').exists()
                    work_status=read_json(work_root/'research/rebuild-status.json',{})
                    requested_work.setdefault('episodes',{}).setdefault('20',{})['stages']={
                        'rebuild':'completed' if rebuilt else 'running',
                        'training':'completed' if trained else 'running' if rebuilt else 'queued',
                        'rollout':'running' if trained else 'queued'}
                    requested_work['reason']='Episode20: '+work_status.get('phase','automatic reconstruction, training and rollout')+'. Original object-motion agreement remains unverified.'
            if requested_work.get('execution_order')==[30,20]:
                plan=human_match_loop.setdefault('execution_plan',{}).setdefault('20',{})
                if requested_work.get('status')=='waiting_episode30':plan['status']='queued_after_episode30'
                elif requested_work.get('status')=='running_episode20':plan['status']='running_corrective_cycle'
        if pickup_repair:
            pickup_repair['report']=artifact(BATCH/'episode30_pickup_repair_20261026/status.json')
            preview=BATCH/'episode30_pickup_repair_20261026/brush-reconstruction-source-review.mp4'
            if preview.is_file():pickup_repair['reconstruction_preview']=artifact(preview)
            pickup_repair['reconstruction_progress']=read_json(Path(pickup_repair['run'])/'research/coupled-reconstruction-progress.json',{})
            try:
                argv=Path(f"/proc/{int(pickup_repair['pid'])}/cmdline").read_bytes()
                pickup_repair['active']=pickup_repair.get('status')=='running' and b'episode30_pickup_repair_20261026.py' in argv
            except (OSError,ValueError,KeyError):pickup_repair['active']=False
            if pickup_repair['active'] and '30' in human_match_loop.get('execution_plan',{}):
                human_match_loop['execution_plan']['30']['status']='running_pickup_repair'
        if human_match_loop:
            human_match_loop['report']=artifact(queue_dir/'status.json')
            human_match_loop['audit_progress']=read_json(queue_dir/'audit-progress.json',{})
        if focus_scope.get('status')=='active':automatic_queue={}
        if automatic_queue:
            automatic_queue['report']=artifact(queue_path)
            automatic_queue['completed_episodes']=[int(ep) for ep,value in automatic_queue.get('episodes',{}).items() if value.get('completed')]
            automatic_queue['needs_repair']=[key for key,value in automatic_queue.get('attempts',{}).items() if value.get('failures',0)>=3]
            if automatic_queue.get('needs_repair'):
                issues.append('Automatic queue exhausted its retries for: '+', '.join(automatic_queue['needs_repair'])+'. Other runnable episodes continue.')
        resources=hardware()
        incomplete=sum(not row['training_complete'] for row in episodes)
        if incomplete and not pids and not issues:
            issues.append('No training process is running. Remaining episodes may be waiting for GPU admission or need queue recovery.')
        phase_name=phase.get('phase','not_started')
        current_trials=[r['latest_trial'] for r in episodes if r.get('latest_trial')]
        reviewing=[r for r in current_trials if r['training_complete'] and not r['export_complete'] and r['status']!='failed']
        needs_motion_review=[r['episode'] for r in episodes if r.get('quality') and not r['quality'].get('all_internal_frame_checks_passed')]
        outdated=[]
        for row in episodes:
            selected=(row.get('videos') or {}).get('best') or {}
            staged=staging.get('episodes',{}).get(str(row['episode']),{})
            if staged.get('status')!='staged_locally':continue
            if (selected.get('checkpoint_sha256') and selected['checkpoint_sha256']!=staged.get('identity',{}).get('checkpoint_sha256') or
                selected.get('recorded_states') and Path(selected['recorded_states']).resolve()!=Path(staged.get('recording','')).resolve()):
                outdated.append(row['episode'])
        if phase_name=='fresh_training' and not pids and not issues and any(not item['training_complete'] for item in fresh):
            issues.append('Fresh retraining is queued, but no learner is active. Check admission and improvement recovery status.')
        hero=('Rechecking all human videos frame by frame.' if human_recheck.get('active') else 'Training is running.' if active else 'Rebuilding the four requested episodes.' if any(r.get('activity_kind')=='complete_reconstruction_and_retraining' for r in active_reviews) else 'Evaluating robot controller corrections.' if active_reviews else 'Testing controller improvements.' if phase_name=='controller_trials' else
              'Testing corresponding-object controllers.' if extra_phase.get('phase')=='actor_combination_trials' else
              'Evaluating and rendering newer rollouts.' if reviewing else
              'Motion improvements need review.' if needs_motion_review else
              'Improvement phase complete.' if phase_name in ['completed','completed_with_experiment_failures'] else
              'Rendering and validating videos.' if batch.get('policies_training_completed')==18 and batch.get('rollout_exports_completed',0)<18 else
              'Baseline videos are complete.' if batch.get('rollout_exports_completed')==18 else 'Waiting for training.')
        round_status=('training' if active else 'evaluating_and_rendering' if reviewing or active_reviews else 'needs_attention' if precision.get('failed') else 'motion_review_required' if needs_motion_review else 'completed')
        user_pause=read_json(BATCH/'user-pause.json',{})
        if user_pause.get('status')=='paused_by_user':hero='Training paused by user.'
        elif human_match_loop.get('active') and not active:
            target=human_match_loop.get('required_episode')
            if human_match_loop.get('active_optimization') or human_match_loop.get('external_optimization'):
                hero=f'Episode {target:06d}: reconstruction or validation; no policy learner active.' if target is not None else 'Reconstruction or validation is running.'
            elif human_match_loop.get('status')=='blocked_by_current_episode_motion':
                hero=f'Episode {target:06d} remains unfinished; repair retry is pending.' if target is not None else 'Motion/contact failures remain; no episode worker is active.'
            else:hero='Checking the current episode; later episodes remain queued.'
        if requested_work.get('status')=='blocked' and not active and not active_reviews:
            hero='Episodes 20 and 30 requested; execution is blocked.'
        if source_cad.get('enabled'):
            cad_ep=source_cad.get('active_episode')
            hero=f'Episode {cad_ep:06d}: CAD and trajectory alignment to original video.' if cad_ep is not None else 'CAD overlays inspected; source alignment failures remain unresolved.'
            if cad_ep is not None and source_cad.get('stage','').startswith('fresh_video_frontend_'):
                frontend_stage=source_cad['stage'].removeprefix('fresh_video_frontend_').replace('_',' ')
                hero=f'Evaluation {cad_ep:06d}: fresh source-video frontend — {frontend_stage}. No evaluation policy is training.'
        if public_validation and (public_validation.get('status')=='running' or human_match_loop.get('status')=='public_pipeline_validation'):
            public_ep=public_validation.get('active_episode')
            public_label=f'Public validation {public_ep:06d}' if public_ep is not None else 'Public validation'
            hero=f'{public_label}: {public_validation.get("phase_detail") or public_validation.get("phase","video reconstruction")}. No evaluation policy is training.'
        experimental_job=human_match_loop.get('active_optimization') or {}
        if str(human_match_loop.get('status','')).startswith('submission_'):
            hero=human_match_loop.get('phase','Official submission replay and packaging. Strict completion unchanged.')
        elif human_match_loop.get('status')=='fast_submission_running':
            from evaluation20_terminal import recorded_steps
            count=recorded_steps(experimental_job['run']) if experimental_job.get('run') else 0
            label=f"Evaluation {experimental_job['episode']:06d}: " if experimental_job.get('episode') is not None else ''
            hero=f"{label}{human_match_loop.get('phase','complete-first submission')}. Actual training steps {count or 0}."
        elif experimental_job.get('experimental_only'):
            experimental_ep=experimental_job['episode']
            hero=f"Evaluation {experimental_ep:06d} experimental pilot: {human_match_loop.get('phase','physical reference preparation')}. Not strict completion."
            if human_match_loop.get('status')=='experimental_policy_training':
                from evaluation20_terminal import recorded_steps
                steps=recorded_steps(experimental_job['run'])
                hero=f"Evaluation {experimental_ep:06d}: experimental FlashSAC training, {steps if steps is not None else 'initializing'} environment steps. Strict episode completion remains gated."
                if (Path(experimental_job['run'])/'flash_chord_coherent_actor_transfer/TRAINING_COMPLETED').exists():
                    hero=f"Evaluation {experimental_ep:06d}: pilot training finished; validating checkpoints and policy rollouts. Strict completion remains gated."
        elif read_json(queue_dir/'experimental-training-status.json',{}).get('status')=='pilot_blocked':
            failed=read_json(queue_dir/'experimental-training-status.json',{})
            failed_root=Path(failed['run']) if failed.get('run') else None
            if failed_root and (failed_root/'flash_chord_coherent_actor_transfer/TRAINING_COMPLETED').exists():
                failed_cfg=read_json(failed_root/'episode-config.json',{})
                hero=f"Evaluation {failed_cfg.get('episode')}: pilot training completed; validation/export blocked. See experimental-training-status.json."
            else:hero+=' Experimental policy preparation failed; see experimental-training-status.json.'
        from full_roster_readiness import update as update_full_roster_review
        full_roster_review=update_full_roster_review(episodes,human_match_loop,staging)
        pipeline_execution=read_json(queue_dir/'all20-pipeline-execution.json',{})
        if pipeline_execution:
            pipeline_execution['report']=artifact(queue_dir/'all20-pipeline-execution.json')
            pipeline_execution['strict_completed']=queue_health.get('episodes_done',[])
            pipeline_execution['public_stage']=public_validation.get('phase')
            pipeline_execution['public_episode']=public_validation.get('active_episode')
            pipeline_execution['policy_training_steps']=queue_health.get('environment_steps',0)
        if read_json(queue_dir/'own-geometry-policy.json',{}).get('enabled'):
            # Historical public-validation bookkeeping is archived evidence;
            # it cannot describe the current evaluation-only reconstruction.
            from evaluation20_terminal import recorded_steps
            own_rows={}
            focus=human_match_loop.get('focused_episodes',[])
            for ep in focus:
                frontend=BASE/f'runs/evaluation_episode_{ep:06d}_frontend_20261029_v4_own_geometry_20261102_v3'
                cfg=read_json(frontend/'episode-config.json',{})
                geometry=read_json(frontend/'research/own-geometry-provenance.json',{})
                poses={name:len(list((frontend/'reconstruction/object_poses'/name).glob('*.json'))) for name in cfg.get('objects',[])}
                own_rows[str(ep)]=dict(input_videos_verified=1 if geometry else 0,
                    new_pipeline_reconstruction=(f"own meshes; native poses {sum(poses.values())}/{cfg.get('frames',0)*len(poses)}" if geometry else 'queued own RGB/depth reconstruction'),
                    policy_training_steps=0,official_policy_rollout='pending',strict_completed=False)
            current_run=Path(experimental_job['run']) if experimental_job.get('run') else None
            if current_run and read_json(current_run/'episode-config.json',{}).get('own_geometry_required'):
                ep=str(read_json(current_run/'episode-config.json',{}).get('episode'))
                if ep in own_rows:own_rows[ep]['policy_training_steps']=recorded_steps(current_run) or 0
            pipeline_execution=dict(updated_utc=human_match_loop.get('updated_utc'),episodes=focus,
                strict_completed=queue_health.get('episodes_done',[]),per_episode=own_rows,
                active_episode=human_match_loop.get('current_episode'),active_stage=source_cad.get('stage') or human_match_loop.get('phase'),
                active_dataset_split='evaluation',public_excluded=True,public_episode=None,
                public_stage='excluded by user; no supplied object scans',
                policy_training_steps=recorded_steps(current_run) if current_run else 0,
                report=artifact(queue_dir/'source-cad-status.json'),submission_ready=False,
                submission_uploaded=read_json(queue_dir/'own-submission-status.json',{}).get('status')=='submitted')
            fast=read_json(queue_dir/'fast-evaluation-status.json',{})
            if fast.get('mode')=='complete_first':
                pipeline_execution.update(complete_first=True,active_stage=human_match_loop.get('phase'),
                    submission_ready_count=sum(r.get('status')=='ready_for_submission' for r in fast.get('episodes',{}).values()),
                    report=artifact(queue_dir/'fast-evaluation-status.json'))
                for ep,row in fast.get('episodes',{}).items():
                    if ep in own_rows and row.get('run'):
                        own_rows[ep].update(policy_training_steps=recorded_steps(row['run']) or 0,
                            official_policy_rollout=row.get('status','pending'))
        from kaggle_progress import current_scores,latest_comparison
        kaggle=current_scores(BATCH)
        comparison=latest_comparison(BATCH)
        from evaluation20_terminal import pilot_validation
        SNAPSHOT={'pilot_validation':pilot_validation(queue_dir),'kaggle_scores':kaggle,'source_pose_repair':source_pose_repair,'full_roster_review':full_roster_review,'training_pause':user_pause,'generated_utc':datetime.now(timezone.utc).isoformat(),'frontend_version':frontend_version(),'source_updated_utc':max([batch.get('updated_utc',''),precision.get('updated_utc','')]+[r.get('updated_utc') or '' for r in current_trials]+[(r.get('videos',{}).get('latest') or {}).get('validated_utc') or '' for r in episodes]+[r.get('updated_utc','') for r in latest_reviews.values()]),
                  'kaggle_comparison':comparison,'refresh_seconds':5,'hero':hero,'episodes':episodes,'active_jobs':active,'active_reviews':active_reviews,'fresh_runs':fresh,'public_validation':public_validation,'pipeline_execution':pipeline_execution,
                  'human_video_recheck':human_recheck,'automatic_queue':automatic_queue,'human_match_loop':human_match_loop,'queue_health':queue_health,'focus_scope':focus_scope,'session_updates':session_updates,'dataset_resource_audit':dataset_resource_audit,'locating_spheres':locating_spheres,'source_cad':source_cad,'collision_validation':collision_validation,'sam3d':sam3d,'pickup_repair':pickup_repair,'requested_work':requested_work,
                  'totals':{'requested':20,'reconstructed':len(episodes),
                            'training_finished':sum(r['training_complete'] for r in episodes),
                            'videos_exported':sum(r['export_complete'] for r in episodes),
                            'videos_verified':sum(r['artifact_verified'] for r in episodes),
                            'baseline_percent':100*sum(r['export_complete'] for r in episodes)/20,
                            'training_percent':sum(r['percent'] for r in episodes)/max(1,len(episodes)),
                            'source_frames_reviewed':sum(r['source_frames'] or 0 for r in episodes),
                            'motion_checks_passed':sum(bool((r.get('quality') or {}).get('all_internal_frame_checks_passed')) for r in episodes),
                            'motion_review_required':len(needs_motion_review)},
                  'improvement':{'phase':phase_name,'details':phase,'user_priority_episodes':priorities,'controller_groups_completed':sum(r.get('status')=='completed' for r in controller_results),
                                 'object_combination_trials':extra_phase,
                                 'grasp_curriculum':grasp_phase,
                                 'backward_grasp':read_json(BATCH/'improvements_20261020_backward/phase-status.json',{}),
                                 'contact_finetuning':contact_phase_status(),
                                 'precision_round':precision,'current_status':round_status,
                                 'current_running_episodes':[r['episode'] for r in active],
                                 'current_reviewing_episodes':sorted({r['episode'] for r in reviewing+active_reviews}),
                                 'motion_review_required_episodes':needs_motion_review,
                                 'controller_groups_total':18,'fresh_runs_complete':sum(r.get('status')=='completed' for r in fresh_results),
                                 'fresh_runs_planned':len(phase.get('episodes_queued',[])),
                                 'measured_improvements':sorted({r['episode'] for r in controller_results+fresh_results+extra_results+grasp_results if r.get('measured_improvement')} |
                                     {r['episode'] for r in episodes if (r.get('videos',{}).get('best') or {}).get('measured_improvement')})},
                  'submission':{'episodes_staged':staging.get('episodes_staged',0),'required':20,
                                'package_ready':staging.get('final_submission_package_ready',False) and full_roster_review.get('ready_for_submission',False),'uploaded':staging.get('uploaded',False),
                                'updated_utc':staging.get('updated_utc'),'outdated_selection_episodes':outdated,
                                'current_selections_staged':not outdated and staging.get('episodes_staged')==20,
                                'remaining_requirements':staging.get('remaining_submission_requirements',[])},
                  'resources':resources,'issues':issues,
                  'links':{'videos':'/videos','improvements':'/improvements',
                           'submission':artifact(BATCH/'submission_staging/readiness.json'),'audit':artifact(BATCH/'all20_evaluation/all20-episode-audit.json'),
                           'object_review':'/object-checks'},
                  'note':'Tracks all20 evaluation episodes. Earlier episodes1 and5 are preserved. Existing deliverables, current improvement trials and successful imitation are reported separately; local motion checks are not official challenge scores.'}
        SNAPSHOT_AT=now
        return SNAPSHOT


class Handler(SimpleHTTPRequestHandler):
    def log_message(self,format,*args):
        if str(args[1] if len(args)>1 else '') not in ['200','206','304']:super().log_message(format,*args)

    def do_GET(self):self.respond(False)
    def do_HEAD(self):self.respond(True)

    def respond(self,head):
        route=urlsplit(self.path).path
        video_match=re.fullmatch(r'/episodes/(\d{6})/(latest|best|baseline)/(comparison|rollout|scene)\.mp4',route)
        if video_match:
            ep=int(video_match[1]);root=RUNS/f'episode_{ep:06d}_newton_batch_20261015'
            release=catalog(ep,root).get(video_match[2])
            location=release.get('links',{}).get(video_match[3]) if release else None
            if not location:self.send_error(404,'No validated video is available');return
            self.send_response(302);self.send_header('Location',location)
            self.send_header('Cache-Control','no-store');self.send_header('Content-Length','0');self.end_headers();return
        if route=='/api/status':
            try:body=json.dumps(collect(),allow_nan=False).encode()
            except Exception as error:
                self.send_error(503,str(error));return
            self.send_response(200);self.send_header('Content-Type','application/json');self.send_header('Cache-Control','no-store')
            self.send_header('Content-Length',str(len(body)));self.end_headers()
            if not head:self.wfile.write(body)
            return
        if route in ['/','/progress.html']:
            path=BASE/'progress_dashboard.html'
        elif route in ['/improvements','/object-checks','/artifacts/evaluation_batch_20261015/improvements_20261019/index.html','/artifacts/evaluation_batch_20261015/improvements_20261019/priority-object-review.html']:
            path=BASE/'live_object_review.html'
        elif route in ['/videos','/artifacts/evaluation_batch_20261015/videos.html']:
            path=BASE/'latest_video_gallery.html'
        elif route=='/human-review':
            path=BASE/'human_video_audit.html'
        elif route.startswith('/artifacts/'):
            path=(RUNS/unquote(route.removeprefix('/artifacts/'))).resolve()
            if not path.is_relative_to(RUNS.resolve()) or path.suffix.lower() not in ['.html','.json','.mp4','.png','.jpg','.jpeg']:
                self.send_error(403);return
        else:self.send_error(404);return
        if not path.is_file():self.send_error(404);return
        if path.suffix.lower()=='.html':
            body=path.read_bytes().replace(b'__FRONTEND_VERSION__',frontend_version().encode())
            self.send_response(200);self.send_header('Content-Type','text/html; charset=utf-8')
            self.send_header('Cache-Control','no-store');self.send_header('Content-Length',str(len(body)));self.end_headers()
            if not head:self.wfile.write(body)
            return
        size=path.stat().st_size;start=0;end=size-1;partial=False
        request=self.headers.get('Range')
        if request:
            match=re.fullmatch(r'bytes=(\d*)-(\d*)',request)
            if not match or not any(match.groups()):self.send_error(416);return
            if match[1]:start=int(match[1]);end=min(size-1,int(match[2])) if match[2] else size-1
            else:start=max(0,size-int(match[2]))
            if start>=size or start>end:self.send_error(416);return
            partial=True
        self.send_response(206 if partial else 200)
        self.send_header('Content-Type',mimetypes.guess_type(path)[0] or 'application/octet-stream')
        self.send_header('Accept-Ranges','bytes');self.send_header('Content-Length',str(end-start+1));self.send_header('Cache-Control','no-cache')
        if partial:self.send_header('Content-Range',f'bytes {start}-{end}/{size}')
        self.end_headers()
        if head:return
        try:
            with path.open('rb') as stream:
                stream.seek(start);remaining=end-start+1
                while remaining:
                    chunk=stream.read(min(65536,remaining))
                    if not chunk:break
                    self.wfile.write(chunk);remaining-=len(chunk)
        except (BrokenPipeError,ConnectionResetError):pass


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('--port',type=int,default=8768);args=parser.parse_args()
    server=ThreadingHTTPServer(('127.0.0.1',args.port),Handler)
    metadata=BATCH/'progress-dashboard-worker.json'
    metadata.write_text(json.dumps({'pid':os.getpid(),'script':Path(__file__).name,
                                   'updated_utc':datetime.now(timezone.utc).isoformat(),
                                   'url':f'http://127.0.0.1:{args.port}/'}))
    print(f'Live Track 3 dashboard: http://127.0.0.1:{args.port}',flush=True)
    server.serve_forever()
