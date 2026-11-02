"""Complete-first mode of the existing serial controller; no additional worker."""
import json
from pathlib import Path
import subprocess
import time
import traceback
from improvement_common import BASE,BATCH,digest,save,utc
from episode_execution_guard import episode_slot

PY=BASE/'runs/episode_000005_newton_20261015/.venv-video/bin/python'


def read(path):
    try:return json.loads(Path(path).read_text())
    except (OSError,ValueError):return {}


def source_estimate_valid(root):
    import numpy as np
    from own_geometry import validate
    root=Path(root);validate(root);cfg=read(root/'episode-config.json')
    path=root/'reconstruction/scene_motion_physical.npz'
    with np.load(path,allow_pickle=False) as data:
        assert data['object_to_world'].shape==(cfg['frames'],len(cfg['objects']),4,4),'Incomplete estimated object trajectory'
        assert data['camera_to_world'].shape==(cfg['frames'],4,4)
        for name in ['object_to_world','camera_to_world','hand_joint_positions']:
            assert np.isfinite(data[name]).all(),('Nonfinite source estimate',name)
    return digest(path)


def run(out,focus):
    from fast_submission_policy import policy
    from source_cad_queue import source_for
    from fresh_evaluation_frontend import run as frontend
    from continuous_human_match_20261023 import next_generation
    from evaluation_experiment_queue import baseline_source,execution_inputs
    from own_submission_queue import eligible,run as submit
    state=read(out/'status.json');progress=read(out/'fast-evaluation-status.json')
    progress.setdefault('episodes',{})
    progress.update(mode='complete_first',strict_completion_granted=False,required_episodes=focus,maximum_active_episodes=1)
    state.update(focused_episodes=focus,expected_episodes=focus,serial_queue_order=focus,maximum_active_episodes=1,active=True)

    def emit(ep,stage,root=None):
        job=dict(episode=ep,run=str(root),action=stage,dataset_split='evaluation',fast_submission=True) if root else None
        state.update(status='fast_submission_running',current_episode=ep,active_optimization=job,phase=stage,updated_utc=utc())
        save(out/'status.json',state)
        progress.update(active_episode=ep,stage=stage,updated_utc=utc());save(out/'fast-evaluation-status.json',progress)

    while True:
        if read(BATCH/'user-pause.json').get('status')=='paused_by_user':return True
        if read(out/'controller-reload-request.json').get('pending'):return True
        for ep in focus:
            if read(BATCH/'user-pause.json').get('status')=='paused_by_user':return True
            if read(out/'controller-reload-request.json').get('pending'):return True
            entry=progress['episodes'].get(str(ep),{})
            if entry.get('run'):
                try:
                    result=eligible(entry['run'])
                    progress['episodes'][str(ep)].update(status='ready_for_submission',trajectory_sha256=result['trajectory_sha256'])
                    continue
                except (OSError,ValueError,KeyError,AssertionError):pass
            try:
                source=source_for(ep)
                initial=BASE/f'runs/evaluation_episode_{ep:06d}_frontend_20261029_v4_own_geometry_20261102_v3'
                with episode_slot(initial,ep):
                    def update(root,stage):emit(ep,'Fast source reconstruction: '+stage,root)
                    root=frontend(source,ep,update)
                    signature=source_estimate_valid(root)
                # Keep the exact same interrupted request/checkpoints. A
                # completed run is never edited into a different experiment.
                request=Path(entry['request']) if entry.get('request') else None
                if request and request.exists():
                    job=read(request)
                    assert job.get('fast_submission') and job.get('source_version')==signature,'Fast request belongs to different source inputs'
                    current=execution_inputs()
                    if job.get('execution_input_hashes')!=current:
                        # An execution repair may resume the same immutable
                        # reference/own learner. Retain the old producing code
                        # and request rather than resetting learned steps.
                        archive=request.with_name(request.stem+'-before-resume-'+digest(request)[:12]+'.json')
                        if not archive.exists():save(archive,job)
                        job.setdefault('execution_resume_code_commits',[]).append(subprocess.check_output(['git','rev-parse','HEAD'],cwd=BASE,text=True).strip())
                        job['execution_input_hashes']=current;save(request,job)
                else:
                    generation=next_generation(ep,1);baseline=baseline_source(source)
                    job=dict(episode=ep,generation=generation,source_run=str(baseline),baseline_run=str(baseline),
                        fast_source_run=str(root),source_version=signature,fast_submission=True,
                        own_geometry_required=True,validated_video_pipeline=True,experimental_training=True,
                        experimental_admission=dict(admitted=True,basis='Finite own-geometry source estimate; quality not required by user',strict_completion_granted=False),
                        public_validation_manifest=str(out/'pipeline_validation_20261029/status.json'),
                        training_budget=int(policy(out).get('training_budget',640000)),own_training_continuation_allowed=False,
                        time_advance_frames=0.,execution_input_hashes=execution_inputs(),
                        producing_code_commit=subprocess.check_output(['git','rev-parse','HEAD'],cwd=BASE,text=True).strip())
                    request=out/f'jobs/episode_{ep:06d}-fast-g{generation:03d}.json';save(request,job)
                run_root=BASE/f"runs/episode_{ep:06d}_newton_human_match_20261023_g{job['generation']:03d}"
                progress['episodes'][str(ep)]=dict(status='running',request=str(request),run=str(run_root))
                emit(ep,'Fast policy preparation, one training pass and one official rollout',run_root)
                logpath=out/f'logs/fast-episode-{ep:06d}-g{job["generation"]:03d}.log';logpath.parent.mkdir(exist_ok=True)
                with logpath.open('a') as log:
                    child=subprocess.Popen([str(PY),'-u',str(BASE/'execute_human_match_job_20261023.py'),'--request',str(request)],
                        cwd=BASE,stdout=log,stderr=subprocess.STDOUT)
                    while child.poll() is None:
                        activity=read(run_root/'research/rebuild-status.json')
                        emit(ep,activity.get('phase','Fast policy preparation/training/evaluation'),run_root);time.sleep(10)
                    if child.returncode:raise RuntimeError(f'Fast episode {ep} failed (exit {child.returncode}); {logpath}')
                result=eligible(run_root)
                progress['episodes'][str(ep)].update(status='ready_for_submission',trajectory_sha256=result['trajectory_sha256'],error=None)
                emit(ep,'Official policy trajectory ready; moving to the next episode',run_root)
                submit(out,focus)
            except Exception as error:
                progress['episodes'].setdefault(str(ep),{}).update(status='execution_error',error=str(error),traceback=traceback.format_exc())
                emit(ep,'Execution error recorded; continue the remaining episodes')
        submit(out,focus)
        if len([v for v in progress['episodes'].values() if v.get('status')=='ready_for_submission'])==len(focus):
            state['submission_cohort_completed']=True
            emit(None,'All 20 valid policy trajectories completed; Kaggle submission/receipt retained')
            return True
        # Incomplete episodes remain explicit. Resume their saved stage outputs
        # after a transient execution error; no quality-only retry/refit loop.
        emit(None,'Retry incomplete execution stages; quality failures do not block submission')
        time.sleep(30)
