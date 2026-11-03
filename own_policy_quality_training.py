"""Source/geometry/contact reconstruction and learned-policy improvement, serially."""
import json,subprocess,time,traceback
from pathlib import Path
from improvement_common import BASE,BATCH,save,utc,digest
from episode_execution_guard import episode_slot

PY=BASE/'runs/episode_000005_newton_20261015/.venv-video/bin/python'


def read(path):
    try:return json.loads(Path(path).read_text())
    except (OSError,ValueError):return {}


def whole_rollout_passed(result,diagnostic):
    return (result.get('local_reference_metrics',{}).get('chord_sr')==1.
        and diagnostic.get('rollout_motion',{}).get('dropped_frames')==0)


def choose_candidate(candidates,baseline_loss):
    supported=[c for c in candidates if whole_rollout_passed(c['result'],c['diagnostic'])
        and c['diagnostic']['source_video_loss']<baseline_loss*.99]
    return min(supported or candidates,key=lambda c:c['diagnostic']['source_video_loss'])


def run(out,focus):
    from own_source_quality import prepare as prepare_source
    from source_policy_validation import score
    from own_submission_queue import eligible,run as submit
    from continuous_human_match_20261023 import next_generation
    from evaluation_experiment_queue import execution_inputs
    from training_continuation import extend
    from own_policy_calibration import diagnose
    from fast_policy_execution import evaluate_checkpoint,select_result
    policy=read(out/'quality-improvement-policy.json');baseline=read(policy['baseline_cohort'])['episodes']
    progress=read(out/'quality-training-status.json');progress.setdefault('episodes',{})
    from quality_followups import apply_repair_requests
    apply_repair_requests(out,progress)
    selection=read(out/'quality-selection.json');selection.setdefault('episodes',{})
    if not progress.get('baseline_selection_initialized'):
        # The exploratory gain-only cohort was not a six-metric improvement.
        # Start geometry/training selection from the preserved valid baseline.
        selection['episodes']={str(ep):dict(run=baseline[str(ep)]['run']) for ep in focus}
        save(out/'quality-selection.json',selection);progress['baseline_selection_initialized']=True
    state=read(out/'status.json');state['submission_cohort_completed']=False
    def emit(ep,phase,root=None):
        state.update(status='quality_improvement_running',phase=phase,current_episode=ep,updated_utc=utc(),
            active_optimization=dict(episode=ep,run=str(root),action=phase,fast_submission=True,quality_improvement=True) if root else None)
        save(out/'status.json',state);progress.update(active_episode=ep,stage=phase,updated_utc=utc());save(out/'quality-training-status.json',progress)
    # The first successful rollout improved MP-SR. Bring the measured geometry
    # candidates forward too, then revisit the early imperfect rollouts.
    order=[35,30,16,28,36,33,20,19,1,25,15,6,3,8,32,14,5,10,37,38]
    order=[ep for ep in order if ep in focus]
    def render_review(ep,root):
        try:
            from quality_policy_overlay import render
            emit(ep,'Render original-source CAD and genuine policy-object overlays',root)
            with episode_slot(root,ep):video=render(root)
            progress.setdefault('overlays',{})[str(ep)]=video['video']
            progress.setdefault('overlay_errors',{}).pop(str(ep),None)
        except Exception as error:progress.setdefault('overlay_errors',{})[str(ep)]=str(error)
        save(out/'quality-training-status.json',progress)
    if not progress.get('source_overlay_backfill_complete'):
        for ep,row in progress['episodes'].items():
            if row.get('status')!='completed' or not row.get('run'):continue
            if read(out/'controller-reload-request.json').get('pending') or read(BATCH/'user-pause.json').get('status')=='paused_by_user':return
            render_review(int(ep),Path(row['run']))
        progress['source_overlay_backfill_complete']=True;save(out/'quality-training-status.json',progress)
    while True:
        for ep in order:
            if read(BATCH/'user-pause.json').get('status')=='paused_by_user':return
            if read(out/'controller-reload-request.json').get('pending'):return
            entry=progress['episodes'].get(str(ep),{})
            if entry.get('status')=='completed' and entry.get('review_revision')==2:continue
            try:
                old=baseline[str(ep)];old_root=Path(old['run']);original_cfg=read(old_root/'episode-config.json');parent=Path(original_cfg['fast_source_run'])
                request=Path(entry['request']) if entry.get('request') else None
                if request and request.exists():
                    # Resume the exact original reconstruction. A later source
                    # algorithm belongs to a fresh job, never a frozen policy.
                    source=Path(read(request)['fast_source_run'])
                    from own_geometry import validate
                    validate(source)
                else:
                    source=BASE/f'runs/evaluation_episode_{ep:06d}_quality_20261103_v2'
                    emit(ep,'Repair source poses, compare multiview surface candidates, and decompose own concave collision surfaces',source)
                    with episode_slot(source,ep):source=prepare_source(parent,ep)
                if request and request.exists():
                    job=read(request)
                    current=execution_inputs()
                    if job['execution_input_hashes']!=current:
                        archive=request.with_name(request.stem+'-before-repair-'+digest(request)[:12]+'.json')
                        if not archive.exists():save(archive,job)
                        job['execution_input_hashes']=current;job.setdefault('execution_resume_code_commits',[]).append(subprocess.check_output(['git','rev-parse','HEAD'],cwd=BASE,text=True).strip());save(request,job)
                else:
                    generation=next_generation(ep,1)
                    job=dict(episode=ep,generation=generation,source_run=str(old_root),baseline_run=str(old_root),fast_source_run=str(source),
                        source_version=digest(source/'reconstruction/scene_motion_physical.npz'),fast_submission=True,own_geometry_required=True,
                        validated_video_pipeline=True,experimental_training=True,experimental_admission=dict(admitted=True,basis='Own multiview geometry and repaired source/contact reference; measured quality candidate',strict_completion_granted=False),
                        public_validation_manifest=str(out/'pipeline_validation_20261029/status.json'),training_budget=policy.get('quality_training_budget',1280000),
                        own_training_continuation_allowed=True,own_training_maximum_steps=policy.get('quality_training_maximum_steps',2560000),
                        quality_training=dict(world_count=128,control_profile=dict(kp=250.,kd=20.,angular_kp=75.)),
                        execution_input_hashes=execution_inputs(),producing_code_commit=subprocess.check_output(['git','rev-parse','HEAD'],cwd=BASE,text=True).strip())
                    request=out/f'jobs/episode_{ep:06d}-quality-g{generation:03d}.json';save(request,job)
                root=BASE/f"runs/episode_{ep:06d}_newton_human_match_20261023_g{job['generation']:03d}"
                progress['episodes'][str(ep)]=dict(status='running',run=str(root),request=str(request));emit(ep,'New physical contact fitting and own-geometry policy training',root)
                candidates=[]
                def analyze(result):
                    diagnostic=dict(score(root,result['trajectory'],parent),rollout_motion=diagnose(result['trajectory']))
                    save(root/f"research/quality-checkpoint-validation/{result['checkpoint_sha256'][:16]}.json",
                        dict(diagnostic,checkpoint_sha256=result['checkpoint_sha256'],actual_training_steps=result['actual_training_steps'],
                            local_reference_success=result.get('local_reference_metrics',{}).get('chord_sr'),
                            trajectory_sha256=result['trajectory_sha256'],source=str(parent)))
                    return diagnostic
                # A controller restart must retain both real evaluated models.
                # In particular, do not exhaust the bounded continuation and
                # then retry the same budget assertion forever.
                for archive in sorted((root/'research/quality-trained-candidates').glob('*.json')):
                    result=read(archive)
                    for key,hash_key in [('checkpoint','checkpoint_sha256'),('trajectory','trajectory_sha256'),('states','states_sha256')]:
                        if key not in result:continue
                        assert digest(Path(result[key]))==result[hash_key],f'Archived quality artifact changed: {archive}'
                    if result.get('trajectory'):
                        candidates.append(dict(result=result,diagnostic=analyze(result)))
                for round_index in range(2):
                    if (read(out/'controller-reload-request.json').get('pending')
                            or read(BATCH/'user-pause.json').get('status')=='paused_by_user'):
                        progress['episodes'][str(ep)]['status']='continuation_pending'
                        emit(ep,'Checkpoint preserved; waiting for the safe serial boundary before continuation',root)
                        return
                    logfile=out/f"logs/quality-training-{ep:06d}-g{job['generation']:03d}-round{round_index}.log";logfile.parent.mkdir(exist_ok=True)
                    with logfile.open('a') as log:
                        child=subprocess.Popen([str(PY),'-u',str(BASE/'execute_human_match_job_20261023.py'),'--request',str(request)],cwd=BASE,stdout=log,stderr=subprocess.STDOUT)
                        while child.poll() is None:
                            emit(ep,read(root/'research/rebuild-status.json').get('phase','Quality policy training/evaluation'),root);time.sleep(10)
                        if child.returncode:raise RuntimeError(f'Quality episode {ep} failed; {logfile}')
                    candidate=eligible(root);result=read(root/'research/fast-submission-result.json');latest_result=result
                    metrics=analyze(result)
                    archive=root/f"research/quality-trained-candidates/{candidate['checkpoint_sha256'][:16]}.json";save(archive,result)
                    candidates.append(dict(result=result,diagnostic=metrics))
                    # Compare actual whole-model recordings at the saved
                    # checkpoints. A later model need not be the best model.
                    known={c['result']['checkpoint_sha256'] for c in candidates}
                    checkpoints=sorted((root/'flash_chord_coherent_actor_transfer').glob('policy_*.safetensors'),key=lambda p:int(p.stem.split('_')[-1]))
                    for checkpoint in checkpoints:
                        if digest(checkpoint) in known:continue
                        emit(ep,'Replay saved whole-policy checkpoints and inspect every source-video frame',root)
                        with episode_slot(root,ep):alternate=evaluate_checkpoint(root,checkpoint,read(root/'episode-config.json')['native_container'])
                        candidates.append(dict(result=alternate,diagnostic=analyze(alternate)))
                        known.add(alternate['checkpoint_sha256'])
                    from quality_optimizer_archive import archive as archive_optimizers
                    archive_optimizers(root)
                    base_metrics=score(old_root,old['trajectory'],parent)
                    best=choose_candidate(candidates,base_metrics['source_video_loss'])
                    if (whole_rollout_passed(best['result'],best['diagnostic'])
                            and best['diagnostic']['source_video_loss']<base_metrics['source_video_loss']*.98):break
                    actual=latest_result['actual_training_steps']
                    maximum=read(root/'episode-config.json').get('own_training_maximum_steps',2560000)
                    if actual>=maximum:break
                    if round_index==0:extend(root,increment=min(1280000,maximum-actual))
                chosen=choose_candidate(candidates,base_metrics['source_video_loss'])
                select_result(root,chosen['result'])
                incumbent=selection['episodes'].get(str(ep),{});incumbent_root=Path(incumbent.get('run',old_root))
                comparison_loss=base_metrics['source_video_loss']
                if incumbent_root.resolve()!=root.resolve():
                    incumbent_result=read(incumbent.get('result_override') or incumbent_root/'research/fast-submission-result.json')
                    if incumbent_result.get('trajectory'):
                        comparison_loss=min(comparison_loss,score(incumbent_root,incumbent_result['trajectory'],parent)['source_video_loss'])
                selected=chosen['diagnostic']['source_video_loss']<comparison_loss*.99
                if selected:selection['episodes'][str(ep)]=dict(run=str(root))
                save(out/'quality-selection.json',dict(selection,updated_utc=utc(),quality_stage='geometry_contacts_training',not_kaggle_improvement_claim=True))
                progress['episodes'][str(ep)]=dict(status='completed',review_revision=2,run=str(root),request=str(request),selected=selected,baseline_source_validation=base_metrics,candidate_source_validation=chosen['diagnostic'])
                render_review(ep,root)
                emit(ep,'Full policy and source-image validation completed; selecting whole episodes only',root)
                reviewed=sum(row.get('status')=='completed' and row.get('review_revision')==2 for row in progress['episodes'].values())
                interval=max(1,int(policy.get('upload_every_reviewed_episodes',4)))
                geometry=read(source/'research/quality-source-completed.json').get('objects',{})
                new_geometry=any(row.get('multiview_geometry_selected') for row in geometry.values())
                decisive=selected and (new_geometry or whole_rollout_passed(chosen['result'],chosen['diagnostic']))
                decisive=decisive and ep not in progress.get('decisive_uploaded_episodes',[])
                milestone=reviewed-progress.get('last_uploaded_reviewed_count',0)>=interval
                upload_count=progress.get('intermediate_upload_count',0)
                if (milestone or decisive) and upload_count<int(policy.get('maximum_intermediate_uploads',4)):
                    emit(ep,'Upload the complete 20-episode cohort at the quality milestone; remote scoring runs independently',root)
                    try:
                        submit(out,focus)
                        if (read(out/'controller-reload-request.json').get('pending')
                                or read(BATCH/'user-pause.json').get('status')=='paused_by_user'):return
                        progress['last_uploaded_reviewed_count']=reviewed;progress.pop('submission_error',None)
                        progress['intermediate_upload_count']=upload_count+1
                        if decisive:progress.setdefault('decisive_uploaded_episodes',[]).append(ep)
                    except Exception as error:
                        progress['submission_error']=str(error)
                    save(out/'quality-training-status.json',progress)
            except Exception as error:
                progress['episodes'].setdefault(str(ep),{}).update(status='execution_error',error=str(error),traceback=traceback.format_exc());emit(ep,'Quality execution needs repair; original valid submission retained')
        if all(progress['episodes'].get(str(ep),{}).get('status')=='completed' for ep in focus):
            emit(None,'All 20 quality experiments validated; packing and submitting selected full policies')
            submit(out,focus)
            progress['cohort_submitted']=True;save(out/'quality-training-status.json',progress)
            from kaggle_quality_comparison import compare
            comparison=compare(out,policy['baseline_receipt'])
            while (comparison.get('metrics') and any(not row.get('valid_scored_comparison') for row in comparison['metrics'].values())):
                if read(BATCH/'user-pause.json').get('status')=='paused_by_user' or read(out/'controller-reload-request.json').get('pending'):return
                receipt=read(comparison['candidate_receipt'])
                if any(row.get('state')=='ERROR' for row in receipt.get('submissions',{}).values()):break
                emit(None,'All 20 policy reviews finished; awaiting the exact submitted Kaggle IDs before choosing follow-ups')
                subprocess.run(['systemctl','--user','start','--no-block','v2d-track3-kaggle-status.service'],check=True)
                time.sleep(30);comparison=compare(out,policy['baseline_receipt'])
            if comparison.get('all_six_improved') and len(focus)==20:
                progress['all_six_metrics_improved']=True;save(out/'quality-training-status.json',progress)
                state['submission_cohort_completed']=True
            elif len(focus)==20:
                from quality_followups import admit
                admitted=admit(out,focus,progress,policy)
                if admitted:
                    emit(None,'Measured source/optimizer follow-ups admitted after all 20 episodes: '+', '.join(map(str,admitted)))
                    continue
            emit(None,'Kaggle results checked: '+('all six metrics improved' if comparison.get('all_six_improved') else 'remaining metric improvements require another measured round'))
            return
        emit(None,'Retry unresolved quality execution stages; trained checkpoints and valid baseline retained');time.sleep(30)
