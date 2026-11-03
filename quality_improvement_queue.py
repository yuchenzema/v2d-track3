"""Measured policy calibration in the existing serial evaluation controller."""
import json,time,traceback
from pathlib import Path
from improvement_common import BASE,BATCH,save,utc
from episode_execution_guard import episode_slot


def read(path):
    try:return json.loads(Path(path).read_text())
    except (OSError,ValueError):return {}


def run(out,focus):
    from own_policy_calibration import derive,evaluate,diagnose
    from own_submission_queue import eligible,run as submit
    source=read(out/'quality-improvement-policy.json')
    baseline=read(source['baseline_cohort'])['episodes']
    progress=read(out/'quality-improvement-status.json');progress.setdefault('episodes',{})
    selection=read(out/'quality-selection.json');selection.setdefault('episodes',{})
    state=read(out/'status.json');state['submission_cohort_completed']=False
    profiles=source.get('control_profiles',[dict(kp=250.,kd=20.,angular_kp=75.),dict(kp=500.,kd=28.,angular_kp=125.)])
    def emit(ep,phase,root=None):
        state.update(status='quality_improvement_running',phase=phase,current_episode=ep,updated_utc=utc(),
            active_optimization=dict(episode=ep,run=str(root),fast_submission=True,quality_improvement=True,action=phase) if root else None)
        save(out/'status.json',state);progress.update(active_episode=ep,stage=phase,updated_utc=utc());save(out/'quality-improvement-status.json',progress)
    for ep in focus:
        if read(BATCH/'user-pause.json').get('status')=='paused_by_user':return
        if read(out/'controller-reload-request.json').get('pending'):return
        row=progress['episodes'].get(str(ep),{})
        if row.get('status')=='control_calibration_complete':continue
        item=baseline[str(ep)];root=Path(item['run']);cfg=read(root/'episode-config.json');parent=Path(item['checkpoint'])
        base=diagnose(item['trajectory']);candidates=[]
        with episode_slot(root,ep):
            for profile in profiles:
                emit(ep,f'Evaluation {ep}: official policy replay with declared wrist calibration {profile}',root)
                try:
                    checkpoint=derive(root,parent,profile)
                    path,result=evaluate(root,checkpoint,cfg['native_container'])
                    candidate=eligible(root,path);candidates.append(dict(selection=dict(run=str(root),result_override=str(path)),
                        diagnostic=result['quality_diagnostic'],checkpoint_sha256=candidate['checkpoint_sha256'],profile=profile))
                except Exception as error:
                    progress['episodes'].setdefault(str(ep),{}).setdefault('errors',[]).append(dict(profile=profile,error=str(error),traceback=traceback.format_exc()))
                    emit(ep,'Control trial failed; retain the valid baseline and diagnose next profile',root)
        acceptable=[c for c in candidates if c['diagnostic']['mean_source_error_cm']<base['mean_source_error_cm']*.98
            and c['diagnostic']['pair_source_error_cm']<=base['pair_source_error_cm']+1.
            and c['diagnostic']['dropped_frames']<=base['dropped_frames']]
        chosen=min(acceptable,key=lambda c:c['diagnostic']['mean_source_error_cm']) if acceptable else None
        selection['episodes'][str(ep)]=chosen['selection'] if chosen else dict(run=str(root))
        selection.update(updated_utc=utc(),quality_stage='control_calibration',local_selection_is_not_kaggle_improvement=True)
        save(out/'quality-selection.json',selection)
        progress['episodes'][str(ep)]=dict(status='control_calibration_complete',baseline_diagnostic=base,candidates=candidates,
            selected_profile=chosen['profile'] if chosen else None,measured_local_motion_improved=bool(chosen))
        emit(ep,'Whole-policy control comparison completed; originals retained',root)
    emit(None,'All 20 control comparisons completed; submitting measured candidate cohort')
    submit(out,focus)
    progress['control_stage_complete']=True;save(out/'quality-improvement-status.json',progress)
    # The next stage is loaded only at a serial boundary. No additional worker
    # is started by this queue or by the GUI/health monitor.
    while read(out/'quality-improvement-policy.json').get('enabled'):
        if read(BATCH/'user-pause.json').get('status')=='paused_by_user':return
        if read(out/'controller-reload-request.json').get('pending'):return
        if (BASE/'own_policy_quality_training.py').is_file():
            from own_policy_quality_training import run as improve_training
            return improve_training(out,focus)
        emit(None,'Control stage completed; preparing source/geometry and training improvements')
        time.sleep(30)
