"""Source/geometry/contact reconstruction and learned-policy improvement, serially."""
import json,subprocess,time,traceback
from pathlib import Path
from improvement_common import BASE,BATCH,save,utc,digest
from episode_execution_guard import episode_slot

PY=BASE/'runs/episode_000005_newton_20261015/.venv-video/bin/python'


def read(path):
    try:return json.loads(Path(path).read_text())
    except (OSError,ValueError):return {}


def run(out,focus):
    from own_source_quality import prepare as prepare_source
    from source_policy_validation import score
    from own_submission_queue import eligible,run as submit
    from continuous_human_match_20261023 import next_generation
    from evaluation_experiment_queue import execution_inputs
    from training_continuation import extend
    policy=read(out/'quality-improvement-policy.json');baseline=read(policy['baseline_cohort'])['episodes']
    progress=read(out/'quality-training-status.json');progress.setdefault('episodes',{})
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
    # After the largest drop, also train the near-success clips early so
    # success rates receive attention alongside average trajectory errors.
    order=[10,37,38,30,16,28,35,36,33,20,19,1,25,15,6,3,8,32,14,5]
    order=[ep for ep in order if ep in focus]
    while True:
        for ep in order:
            if read(BATCH/'user-pause.json').get('status')=='paused_by_user':return
            if read(out/'controller-reload-request.json').get('pending'):return
            entry=progress['episodes'].get(str(ep),{})
            if entry.get('status')=='completed':continue
            try:
                old=baseline[str(ep)];old_root=Path(old['run']);original_cfg=read(old_root/'episode-config.json');parent=Path(original_cfg['fast_source_run'])
                source=BASE/f'runs/evaluation_episode_{ep:06d}_quality_20261103_v1'
                emit(ep,'Repair source poses, fuse independent depth views, and decompose own concave collision surfaces',source)
                with episode_slot(source,ep):source=prepare_source(parent,ep)
                request=Path(entry['request']) if entry.get('request') else None
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
                # A controller restart must retain both real evaluated models.
                # In particular, do not exhaust the bounded continuation and
                # then retry the same budget assertion forever.
                for archive in sorted((root/'research/quality-trained-candidates').glob('*.json')):
                    result=read(archive)
                    for key,hash_key in [('checkpoint','checkpoint_sha256'),('trajectory','trajectory_sha256'),('states','states_sha256')]:
                        if key not in result:continue
                        assert digest(Path(result[key]))==result[hash_key],f'Archived quality artifact changed: {archive}'
                    if result.get('trajectory'):
                        candidates.append(dict(result=result,diagnostic=score(root,result['trajectory'],parent)))
                for round_index in range(2):
                    logfile=out/f"logs/quality-training-{ep:06d}-g{job['generation']:03d}-round{round_index}.log";logfile.parent.mkdir(exist_ok=True)
                    with logfile.open('a') as log:
                        child=subprocess.Popen([str(PY),'-u',str(BASE/'execute_human_match_job_20261023.py'),'--request',str(request)],cwd=BASE,stdout=log,stderr=subprocess.STDOUT)
                        while child.poll() is None:
                            emit(ep,read(root/'research/rebuild-status.json').get('phase','Quality policy training/evaluation'),root);time.sleep(10)
                        if child.returncode:raise RuntimeError(f'Quality episode {ep} failed; {logfile}')
                    candidate=eligible(root);metrics=score(root,candidate['trajectory'],parent);result=read(root/'research/fast-submission-result.json')
                    archive=root/f"research/quality-trained-candidates/{candidate['checkpoint_sha256'][:16]}.json";save(archive,result)
                    candidates.append(dict(result=result,diagnostic=metrics))
                    from quality_optimizer_archive import archive as archive_optimizers
                    archive_optimizers(root)
                    base_metrics=score(old_root,old['trajectory'],parent)
                    if metrics['source_video_loss']<base_metrics['source_video_loss']*.98:break
                    actual=result['actual_training_steps']
                    maximum=read(root/'episode-config.json').get('own_training_maximum_steps',2560000)
                    if actual>=maximum:break
                    if round_index==0:extend(root,increment=min(1280000,maximum-actual))
                chosen=min(candidates,key=lambda item:item['diagnostic']['source_video_loss'])
                save(root/'research/fast-submission-result.json',chosen['result'])
                selected=chosen['diagnostic']['source_video_loss']<base_metrics['source_video_loss']*.99
                if selected:selection['episodes'][str(ep)]=dict(run=str(root))
                save(out/'quality-selection.json',dict(selection,updated_utc=utc(),quality_stage='geometry_contacts_training',not_kaggle_improvement_claim=True))
                progress['episodes'][str(ep)]=dict(status='completed',run=str(root),request=str(request),selected=selected,baseline_source_validation=base_metrics,candidate_source_validation=chosen['diagnostic'])
                emit(ep,'Full policy and source-image validation completed; selecting whole episodes only',root)
            except Exception as error:
                progress['episodes'].setdefault(str(ep),{}).update(status='execution_error',error=str(error),traceback=traceback.format_exc());emit(ep,'Quality execution needs repair; original valid submission retained')
        if all(progress['episodes'].get(str(ep),{}).get('status')=='completed' for ep in focus):
            emit(None,'All 20 quality experiments validated; packing and submitting selected full policies')
            submit(out,focus)
            progress['cohort_submitted']=True;save(out/'quality-training-status.json',progress)
            from kaggle_quality_comparison import compare
            comparison=compare(out,policy['baseline_receipt'])
            emit(None,'Kaggle results checked: '+('all six metrics improved' if comparison.get('all_six_improved') else 'remaining metric improvements require another measured round'))
            return
        emit(None,'Retry unresolved quality execution stages; trained checkpoints and valid baseline retained');time.sleep(30)
