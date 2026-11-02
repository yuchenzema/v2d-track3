"""Test fixed per-hand actor combinations for the eight unresolved source videos.

Both actors receive the same current physical state. A complete continuous
attempt is evaluated, with zero object assistance and no frame switching.
Original policies and frozen training inputs are preserved.
"""
from pathlib import Path
import argparse
import fcntl
import json
import shutil
import time
import traceback
from improvement_common import BASE,BATCH,OUT as FIRST,baseline,root,better,digest,save,utc,verify_preserved
from batch_review import review
from batch_native import docker
from render_improved_controller import render

OUT=BATCH/'improvements_20261020'
PRIORITY=[20,19,32,33,30,15,10,16]
PROFILES=[{'left':0.,'right':1.},{'left':1.,'right':0.},{'left':.5,'right':.5}]


def phase(name,**kwargs):
    save(OUT/'phase-status.json',{'phase':name,'updated_utc':utc(),'episodes':PRIORITY,
         'method':'Fixed hand weights on two learned policies; same live physical observation',
         'object_assistance':0.,'source_time_warping':False,'original_results_preserved':True,**kwargs})


def context(ep):
    fresh=json.loads((FIRST/f'episode_{ep:06d}-fresh-result.json').read_text())
    if fresh.get('status')!='completed' or not fresh.get('artifact_integrity',{}).get('integrity_passed'):
        raise ValueError('Wait for a complete and verified fresh experiment')
    r=Path(fresh['run']);cfg=json.loads((r/'episode-config.json').read_text());original=baseline(ep)
    directory=r/'research/actor_combination_20261020';directory.mkdir(exist_ok=True)
    source=root(ep)/original['checkpoint'].removeprefix('/episode/')
    alternate=directory/'alternate-baseline-policy.safetensors'
    if not alternate.exists():shutil.copy2(source,alternate)
    if digest(alternate)!=digest(source):raise ValueError('Alternate policy differs from the preserved baseline')
    return r,cfg,fresh,original,alternate


def trial(ep,profile):
    r,cfg,fresh,original,alternate=context(ep);primary=fresh['candidate']
    spec={'alternate_checkpoint':'/episode/'+str(alternate.relative_to(r)),
          'alternate_checkpoint_sha256':digest(alternate),'hand_weights':profile}
    return review(ep,primary['environment_steps'],primary['wrist_reference_velocity_gain'],
                  primary['inference_reference'],primary['physics_substeps'],
                  primary.get('robot_release_controller_spec'),checkpoint_override=primary['checkpoint'],
                  root_override=r,container_override=cfg['native_container'],actor_combination=spec)


def optimize(ep):
    path=OUT/f'episode_{ep:06d}-combination-result.json'
    if path.exists() and json.loads(path.read_text()).get('status')=='completed':return json.loads(path.read_text())
    r,cfg,fresh,original,alternate=context(ep)
    previous=fresh['candidate'] if fresh.get('measured_improvement') else original
    candidates=[];errors=[]
    for number,profile in enumerate(PROFILES,1):
        phase('actor_combination_trials',episode=ep,trial=number,trials_per_episode=len(PROFILES),hand_weights=profile)
        try:candidates.append(trial(ep,profile))
        except Exception as error:
            errors.append({'hand_weights':profile,'error':str(error),'traceback':traceback.format_exc()})
        save(path,{'episode':ep,'status':'testing','updated_utc':utc(),'candidates':candidates,'errors':errors})
    if not candidates:raise RuntimeError(f'Every actor-combination trial failed: {errors}')
    eligible=sorted([row for row in candidates if better(row,previous)],
                    key=lambda row:row['object_motion_quality']['score'],reverse=True)
    selected=None;rendered=None;cohort=None;quality=None
    previous_sr=fresh['candidate_128_world_completion'] if fresh.get('measured_improvement') else fresh['baseline_128_world_completion']
    for row in eligible[:2]:
        import hashlib
        tag=hashlib.sha256(json.dumps(row['actor_combination'],sort_keys=True).encode()).hexdigest()[:10]
        destination=r/'research/actor_combination_20261020'/tag;destination.mkdir(exist_ok=True)
        metrics=destination/'metrics-128.json'
        try:
            if not metrics.exists():
                docker(['exec',cfg['native_container'],'env',
                        f'V2D_ACTOR_COMBINATION={json.dumps(row["actor_combination"],sort_keys=True)}',
                        f'V2D_WRIST_VELOCITY_GAIN={row["wrist_reference_velocity_gain"]}',
                        f'V2D_SIM_SUBSTEPS={row["physics_substeps"]}',
                        f"V2D_REFERENCE_OVERRIDE=/episode/human_motion_data/{row['inference_reference']}/processed/sequence_id={cfg['sequence']}/robot_name=sharpa_wave",
                        *([f'V2D_RELEASE_SPEC={json.dumps(row["robot_release_controller_spec"])}'] if row.get('robot_release_controller_spec') else []),
                        'python','-u','/episode/scripts/evaluate_actor_combination.py',
                        f'evaluation.checkpoint={row["checkpoint"]}','evaluation.world_count=128','logging.mode=disabled',
                        'evaluation.metrics_output=/episode/'+str(metrics.relative_to(r)),
                        f'hydra.run.dir=/episode/hydra/actor-combination-{tag}-128'],r/f'logs/actor-combination-{tag}-128.log')
            cohort=json.loads(metrics.read_text())
            if cohort['metrics']['chord_sr']<previous_sr-.0625:
                errors.append({'hand_weights':row['actor_combination']['hand_weights'],'rejected':'128-world completion regressed'});continue
            rendered=render(ep,row,destination,root_override=r,container_override=cfg['native_container'])
            selected=row
            quality={'all_internal_frame_checks_passed':row['object_motion_quality']['every_object_passes_all_local_frames'] and cohort['metrics']['chord_sr']==1.,
                     'completion_rate_against_video_estimate':cohort['metrics']['chord_sr'],
                     'source_frames_below_internal_0_5_iou':{n:v['source_frames_below_0_5_iou'] for n,v in row['object_motion_quality']['objects'].items()},
                     'full_human_video_match_verified':False}
            break
        except Exception as error:errors.append({'error':str(error),'traceback':traceback.format_exc()})
    verify_preserved(ep)
    result={'episode':ep,'status':'completed','updated_utc':utc(),'evaluation_run':str(r),
            'experiment':'Fixed per-hand combination of baseline and fresh learned actors',
            'baseline':original,'previous_best':previous,'selected':selected or previous,
            'candidates':candidates,'errors':errors,'measured_improvement':selected is not None,
            'rendered':rendered,'native_128_world_evaluation':cohort,'quality_assessment':quality,
            'full_human_motion_match_verified':False}
    save(path,result);print(ep,'fixed hand-policy combinations complete; improved',selected is not None,flush=True)
    return result


def watch():
    results={}
    while len(results)<len(PRIORITY):
        available=[]
        for ep in PRIORITY:
            if ep in results:continue
            p=FIRST/f'episode_{ep:06d}-fresh-result.json'
            if p.exists():
                fresh_status=json.loads(p.read_text()).get('status')
                if fresh_status=='completed':available.append(ep)
                elif fresh_status=='failed':
                    result={'episode':ep,'status':'failed','updated_utc':utc(),'error':'The required fresh-policy experiment failed; no unverified actor was used'}
                    save(OUT/f'episode_{ep:06d}-combination-result.json',result);results[ep]=result
        if not available:
            phase('waiting_for_verified_policies',episodes_completed=list(results));time.sleep(30);continue
        for ep in available:
            try:results[ep]=optimize(ep)
            except Exception as error:
                result={'episode':ep,'status':'failed','updated_utc':utc(),'error':str(error),'traceback':traceback.format_exc()}
                save(OUT/f'episode_{ep:06d}-combination-result.json',result);results[ep]=result
    phase('completed',episodes_with_measured_improvements=[ep for ep,row in results.items() if row.get('measured_improvement')],
          failed_experiments=[ep for ep,row in results.items() if row['status']=='failed'],
          exact_human_motion_match_verified=False)


if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('--preflight',type=int);args=parser.parse_args();OUT.mkdir(exist_ok=True)
    with (OUT/'queue.lock').open('a') as lock:
        fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
        if args.preflight:
            row=trial(args.preflight,PROFILES[0]);save(OUT/f'episode_{args.preflight:06d}-preflight.json',row)
            print(json.dumps({k:row[k] for k in ['completion_rate_32_worlds','source_mean_object_iou','single_attempt_tracking']},indent=2))
        else:watch()
