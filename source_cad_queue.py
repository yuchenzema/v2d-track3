"""Durable, serial CAD/trajectory inspection within the existing queue service."""
import json
import os
from pathlib import Path
import subprocess
import time
import traceback

from improvement_common import BASE,BATCH,save,utc,digest
from episode_execution_guard import episode_slot,register_episode_child

PY=BASE/'runs/episode_000005_newton_20261015/.venv-video/bin/python'

def read(path,default=None):
    try:return json.loads(path.read_text())
    except (OSError,ValueError):return default


def changed_repair_inputs(report, last_attempt=None):
    """A real algorithm change can justify another attempt after the cap.

    Compare with the last attempted input, including an execution failure, so
    an unchanged broken command cannot use a stale report for unlimited retries.
    """
    previous = last_attempt if last_attempt is not None else report.get('algorithm_files_sha256', {})
    return [name for name, old in previous.items()
            if Path(name).name == name and (BASE/name).is_file() and digest(BASE/name) != old]


def algorithm_snapshot(report):
    return {name: digest(BASE/name) for name in report.get('algorithm_files_sha256', {})
            if Path(name).name == name and (BASE/name).is_file()}


def source_for(ep):
    candidates=[r for r in (BASE/'runs').glob(f'episode_{ep:06d}_newton*')
                if (r/'episode-config.json').is_file()
                and (r/'reconstruction/scene_motion_physical.npz').is_file()
                and (r/'reconstruction/frames').is_dir()
                and (r/'reconstruction/masks/1').is_dir()]
    # Reuse the best image reconstruction as initialization; no old policy or
    # physics target is adopted as a measurement of human motion.
    def score(r):
        quality=read(r/'evaluation/reference-source-fidelity.json',{}).get('aggregate',{}).get('objects',{})
        return (sum(o.get('iou',0) for o in quality.values())/max(1,len(quality)),
                (r/'reconstruction/scene_motion_physical.npz').stat().st_mtime)
    assert candidates,('No reconstructed source inputs',ep)
    return max(candidates,key=score)


def run(out,focus,coverage_only=False):
    path=out/'source-cad-status.json';cad=read(path,{})
    validated=read(out/'validated-pipeline-policy.json',{}).get('enabled',False)
    version='fresh_stereo_sift_frontend_v4' if read(out/'validated-pipeline-policy.json',{}).get('camera_odometry_method')=='sift_pnp_metric' else 'depth_markers_source_clock_v2'
    if validated and cad.get('pipeline_version')!=version:
        if cad:save(out/f'source-cad-status-before-validated-{int(time.time())}.json',cad)
        cad=dict(pipeline_version=version,results={},attempts={})
    cad.update(enabled=True,episodes=focus,maximum_active_episodes=1,
               phase_order=['CAD_geometry','source_frame_pose_fitting','CAD_overlay_and_trajectory_audit','physical_retargeting','policy_training','rollout'],
               episode_done=False,strict_completion_count=0,
               policy_training_allowed=False,
               reconstruction_audits_only=coverage_only,
               inspection_is_not_episode_completion=True)
    cad.setdefault('results',{});cad.setdefault('attempts',{})
    state=read(out/'status.json',{})
    state.update(active=True,focused_episodes=focus,expected_episodes=focus,
                 serial_queue_order=focus,maximum_active_episodes=1,
                 execution_mode='one_episode_at_a_time',episodes_done=[],pending_episodes=focus,
                 phase='CAD reconstruction and source trajectory alignment before robot policy training')
    def emit():
        cad.update(updated_utc=utc(),pid=os.getpid());save(path,cad)
        state.update(updated_utc=utc(),pid=os.getpid(),source_cad_first=True,
                     source_cad_progress=str(path),exact_human_match_verified=False)
        save(out/'status.json',state)
    while True:
        if read(BATCH/'user-pause.json',{}).get('status')=='paused_by_user':
            state.update(status='paused_by_user',active=False,current_episode=None,active_optimization=None)
            cad.update(status='paused_by_user',active_episode=None);emit();return
        if read(out/'controller-reload-request.json',{}).get('pending'):
            cad.update(status='awaiting_safe_controller_reload',active_episode=None,
                       policy_training_allowed=False)
            state.update(status='awaiting_safe_controller_reload',current_episode=None,active_optimization=None)
            emit();return False
        if not coverage_only:
            experimental=read(out/'experimental-training-policy.json',{})
            if experimental.get('enabled'):
                from evaluation_experiment_queue import candidates as admitted_pilots
                if admitted_pilots(out,focus,experimental):
                    cad.update(status='experimental_candidate_ready',active_episode=None,policy_training_allowed=False)
                    state.update(status='experimental_candidate_ready',current_episode=None,active_optimization=None)
                    emit();return False
        pending=[ep for ep in focus if str(ep) not in cad['results']]
        repair=False
        if not pending:
            if coverage_only:
                cad.update(status='all_selected_overlays_inspected_training_still_gated',active_episode=None,
                           inspected_count=len(cad['results']),policy_training_allowed=False,
                           episode_done=False,strict_completion_count=0)
                state.update(status='evaluation_source_audits_complete_training_gated',current_episode=None,
                             active_optimization=None)
                emit();return False
            if all(r.get('local_overlap_failed_object_frames')==0 and r.get('unobservable_object_frames')==0
                   for r in cad['results'].values()):
                cad.update(status='all_source_cad_overlays_passed_local_checks',active_episode=None,
                           policy_training_allowed=True,exact_human_motion_verified=False)
                state.update(active_optimization=None,current_episode=None);emit();return True
            # Inspect every queued clip before declaring this pass complete.
            # Failed/hidden frames remain blocked; never advance robot training
            # on the strength of an exported overlay or aggregate mean IoU.
            cad.update(status='all_selected_overlays_inspected_alignment_repairs_required',active_episode=None,
                       inspected_count=len(cad['results']),
                       local_overlap_pass_count=sum(x.get('local_overlap_failed_object_frames')==0 for x in cad['results'].values()))
            candidates=[ep for ep in focus if cad['results'][str(ep)].get('local_overlap_failed_object_frames',0)>0
                        and (cad['attempts'].get(str(ep),0)<3 or changed_repair_inputs(
                            cad['results'][str(ep)],cad.get('last_attempt_algorithm_hashes',{}).get(str(ep))))]
            if not candidates:
                state.update(status='source_alignment_repairs_required',current_episode=None,active_optimization=None)
                emit();time.sleep(30);continue
            # Work first on the weakest object of episodes closest to pilot
            # admission; aggregate IoU can hide a badly aligned second object.
            from experimental_training_admission import assess
            pilot_policy=read(out/'experimental-training-policy.json',{})
            readiness={ep:assess(cad['results'][str(ep)],pilot_policy)['minimum_qualified_fraction']
                       for ep in candidates} if pilot_policy.get('enabled') else {}
            pending=sorted(candidates,key=lambda ep:(-readiness.get(ep,0),cad['attempts'].get(str(ep),0),focus.index(ep)))
            repair=True
        ep=pending[0];key=str(ep)
        source=Path(cad.setdefault('sources',{}).setdefault(key,str(source_for(ep))))
        root=BASE/f'runs/episode_{ep:06d}_cad_source_{"validated_v4" if version=="fresh_stereo_sift_frontend_v4" else "validated" if validated else "alignment"}_20261029'
        previous_failure=cad.pop('latest_failure',None)
        if previous_failure:
            cad['previous_failure']=dict(episode=ep,message=previous_failure,archived_utc=utc())
        cad.update(status='running',active_episode=ep,run=str(root),stage='CAD_geometry_and_source_pose_fit')
        if repair:
            report=cad['results'][key]
            cad['repair_input_changes']=changed_repair_inputs(report,cad.get('last_attempt_algorithm_hashes',{}).get(key))
            cad.setdefault('last_attempt_algorithm_hashes',{})[key]=algorithm_snapshot(report)
        cad['attempts'][key]=cad['attempts'].get(key,0)+1
        state.update(status='source_cad_alignment',required_episode=focus[0],current_episode=ep,
                     active_optimization={'episode':ep,'run':str(root),'pid':os.getpid(),'action':'CAD alignment to original human video','started_utc':utc()})
        for n,row in state.get('execution_plan',{}).items():
            if int(n) in focus:
                row.update(status='running_source_cad_alignment' if int(n)==ep else 'source_cad_inspected' if n in cad['results'] else 'queued_source_cad_alignment',
                           stages=cad['phase_order'],episode_done=False)
        emit()
        try:
            with episode_slot(root,ep):
                if version=='fresh_stereo_sift_frontend_v4':
                    from fresh_evaluation_frontend import run as rebuild_frontend
                    def frontend_progress(frontend,stage):
                        cad.update(stage=f'fresh_video_frontend_{stage}',run=str(frontend))
                        state.update(status='evaluation_frontend',active_optimization={'episode':ep,'run':str(frontend),'pid':os.getpid(),'action':f'fresh video frontend: {stage}','dataset_split':'evaluation'})
                        emit()
                    source=rebuild_frontend(source,ep,frontend_progress)
                    cad['sources'][key]=str(source);cad.update(stage='CAD_geometry_and_source_pose_fit',run=str(root))
                    state.update(status='source_cad_alignment',active_optimization={
                        'episode':ep,'run':str(root),'pid':os.getpid(),
                        'action':'CAD alignment to original human video','dataset_split':'evaluation',
                        'started_utc':utc()})
                    emit()
                root.mkdir(exist_ok=True);(root/'logs').mkdir(exist_ok=True)
                with (root/'logs/source-cad-alignment.log').open('a') as log:
                    command=[str(PY),'-u',str(BASE/'source_cad_alignment.py'),'--source',str(source),'--run',str(root),
                             '--maxfev',str(100 if not repair else 180*cad['attempts'][key])]
                    if repair:command.append('--repair-only')
                    child=subprocess.Popen(command,
                                           cwd=BASE,stdout=log,stderr=subprocess.STDOUT)
                    register_episode_child(child.pid)
                    while child.poll() is None:
                        cad['progress']=read(root/'research/cad-progress.json',{})
                        emit();time.sleep(15)
                    if child.returncode:raise RuntimeError(f'CAD source fit exited {child.returncode}; see {root}/logs/source-cad-alignment.log')
                report=read(root/'research/cad-alignment-report.json',{})
                assert report.get('all_source_frames_checked') and Path(report['overlay']).is_file()
                cad['results'][key]=report
                cad['inspected_count']=len(cad['results'])
                cad.pop('latest_failure',None);emit()
        except Exception:
            cad.update(status='source_alignment_failed',latest_failure=traceback.format_exc())
            emit();time.sleep(60)

