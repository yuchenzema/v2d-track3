"""After all baseline videos finish, test bounded controllers and fresh policies.

Every old result is preserved. Candidates must demonstrate a measured gain on
whole, unassisted physical attempts before they are labeled improvements.
"""
from pathlib import Path
import concurrent.futures
import fcntl
import hashlib
import html
import json
import os
import shutil
import subprocess
import sys
import time
import traceback
from improvement_common import (BASE,BATCH,OUT,EPISODES,root,save,utc,baseline_ready,digest,
                                baseline,preserve,verify_preserved,better)
from batch_review import review
from batch_native import containers,docker
from render_improved_controller import render
from fresh_improvement_run import run as fresh_run

USER_PRIORITY=[20,19,32,33,30,15,10,16]
PRIORITY=USER_PRIORITY+[8,25,36,28,3,6,14,35,37,38]
PROFILES=[{'wrist_position':.5,'wrist_orientation':.5,'finger':1.},
          {'wrist_position':.5,'wrist_orientation':1.,'finger':1.},
          {'wrist_position':.75,'wrist_orientation':.75,'finger':1.},
          {'wrist_position':.5,'wrist_orientation':.5,'finger':.75},
          {'wrist_position':.25,'wrist_orientation':.5,'finger':1.}]


def gallery(state):
    cards=[]
    for ep in EPISODES:
        profile=OUT/f'episode_{ep:06d}-controller-result.json'
        fresh=OUT/f'episode_{ep:06d}-fresh-result.json'
        rows=[json.loads(p.read_text()) for p in [profile,fresh] if p.exists()]
        labels=[];links=[]
        for row in rows:
            kind='Fresh training' if 'experiment' in row else 'Robot action gains'
            labels.append(f'{kind}: {row["status"]}; measured improvement: {row.get("measured_improvement",False)}')
            if row.get('rendered'):
                path=Path(row['rendered']['validated_videos'][-1]['path'])
                links.append(f'<a href="{html.escape(os.path.relpath(path,OUT))}">Original / baseline / improved policy</a>')
            if row.get('run') and row.get('status')=='completed':
                path=Path(row['run'])/'videos/original_vs_policy.mp4'
                links.append(f'<a href="{html.escape(os.path.relpath(path,OUT))}">Fresh-policy comparison</a>')
        baseline_link=html.escape(os.path.relpath(root(ep)/'videos/original_vs_policy.mp4',OUT))
        image=OUT/f'episode_{ep:06d}-failure-review.jpg'
        body=f'<article><h2>Episode {ep:06d}</h2><p>{html.escape(" | ".join(labels) or "Waiting for improvement phase")}</p><p><a href="{baseline_link}">Preserved baseline video</a>'
        if image.exists():body+=f' · <a href="{image.name}">Source-frame failure review</a>'
        body+='</p><p>'+' · '.join(links)+'</p></article>';cards.append(body)
    page='''<!doctype html><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1"><meta http-equiv="refresh" content="60"><title>Track 3 improvement comparisons</title><style>body{font:16px system-ui;background:#10141c;color:#eee;max-width:1100px;margin:auto;padding:24px}a{color:#9bd2ff}article{border:1px solid #465366;padding:16px;margin:14px 0;border-radius:8px}p{line-height:1.6}</style>'''
    page+=f'<h1>Track 3 improvement comparisons</h1><p>Phase: <b>{html.escape(state["phase"])}</b> · Updated {state["updated_utc"]}</p><p>Baseline results stay unchanged. Every candidate uses continuous Newton physics with zero object guidance and no failure resets. Local checks do not establish exact human imitation or an official challenge score. “Measured improvement” requires the declared source-agreement and tracking criteria.</p><p><a href="priority-object-review.html">Priority episodes: corresponding-object frame review</a> · <a href="phase-status.json">Detailed phase status</a> · <a href="failure-triage.json">Failure analysis</a></p>'+''.join(cards)
    temporary=OUT/'index.html.tmp';temporary.write_text(page);temporary.replace(OUT/'index.html')


def phase(name,**kwargs):
    value={'phase':name,'updated_utc':utc(),'baseline_episodes_required':18,
           'baseline_artifacts_overwritten':False,'exact_human_motion_match_verified':False,**kwargs}
    save(OUT/'phase-status.json',value);gallery(value)


def profile_trials(ep):
    r=root(ep);path=OUT/f'episode_{ep:06d}-controller-result.json'
    if path.exists() and json.loads(path.read_text()).get('status')=='completed':return json.loads(path.read_text())
    preserve(ep);original=baseline(ep)
    original_report=json.loads((r/'result-report.json').read_text())
    if original_report['quality_assessment'].get('all_internal_frame_checks_passed'):
        result={'episode':ep,'status':'completed','measured_improvement':False,
                'reason':'Baseline already passes all internal frame checks; preserve its successful controller.',
                'baseline':original,'candidates':[]};save(path,result);return result
    candidates=[];errors=[]
    for number,profile in enumerate(PROFILES,1):
        phase('controller_trials',episode=ep,trial=number,trials_per_episode=len(PROFILES),residual_gain_profile=profile)
        row=None
        for attempt in range(2):
            try:
                row=review(ep,original['environment_steps'],original['wrist_reference_velocity_gain'],
                           original.get('inference_reference'),original.get('physics_substeps'),
                           original.get('robot_release_controller_spec'),checkpoint_override=original['checkpoint'],
                           residual_profile=profile)
                break
            except Exception as error:
                errors.append({'profile':profile,'attempt':attempt+1,'error':str(error),'traceback':traceback.format_exc()})
        if row is not None:candidates.append(row)
        save(path,{'episode':ep,'status':'testing','baseline':original,'candidates':candidates,'errors':errors})
    eligible=sorted([row for row in candidates if better(row,original)],key=lambda row:row['source_agreement_score'],reverse=True)
    chosen=None;rendered=None;cohort=None
    for candidate in eligible[:2]:
        tag=hashlib.sha256(json.dumps(candidate['residual_gain_profile'],sort_keys=True).encode()).hexdigest()[:8]
        directory=r/'research/improvement_20261019'/tag;directory.mkdir(parents=True,exist_ok=True)
        output='/episode/'+str((directory/'metrics-128.json').relative_to(r))
        container=containers(r,ep)[0]
        try:
            if not (directory/'metrics-128.json').exists():
                docker(['exec',container,'env',f'V2D_RESIDUAL_PROFILE={json.dumps(candidate["residual_gain_profile"],sort_keys=True)}',
                        f'V2D_WRIST_VELOCITY_GAIN={candidate["wrist_reference_velocity_gain"]}',
                        *([f'V2D_RELEASE_SPEC={json.dumps(candidate["robot_release_controller_spec"])}'] if candidate.get('robot_release_controller_spec') else []),
                        f'V2D_SIM_SUBSTEPS={candidate["physics_substeps"]}',
                        f"V2D_REFERENCE_OVERRIDE=/episode/human_motion_data/{candidate['inference_reference']}/processed/sequence_id=episode_{ep:06d}/robot_name=sharpa_wave",
                        'python','-u','/episode/scripts/evaluate_residual_profile.py',f'evaluation.checkpoint={candidate["checkpoint"]}',
                        'evaluation.world_count=128','logging.mode=disabled',f'evaluation.metrics_output={output}',
                        f'hydra.run.dir=/episode/hydra/improved-{tag}-128'],r/f'logs/improved-{tag}-128.log')
            cohort=json.loads((directory/'metrics-128.json').read_text())
            if cohort['metrics']['chord_sr']<original_report['quality_assessment']['completion_rate_against_video_estimate']-.0625:
                errors.append({'profile':candidate['residual_gain_profile'],'rejected':'128-world completion regressed beyond the declared allowance'})
                continue
            rendered=render(ep,candidate,directory);chosen=candidate;break
        except Exception as error:
            errors.append({'profile':candidate['residual_gain_profile'],'error':str(error),'traceback':traceback.format_exc()})
    verify_preserved(ep)
    result={'episode':ep,'status':'completed','updated_utc':utc(),'measured_improvement':chosen is not None,
            'baseline':original,'selected':chosen or original,'candidates':candidates,'errors':errors,
            'rendered':rendered,'native_128_world_evaluation':cohort,
            'baseline_128_world_completion':original_report['quality_assessment']['completion_rate_against_video_estimate'],
            'baseline_artifacts_preserved_and_hash_verified':True,'full_human_motion_match_verified':False}
    save(path,result);print(ep,'controller trials complete; measured improvement',chosen is not None,flush=True)
    return result


def run_fresh_with_retry(ep):
    existing=OUT/f'episode_{ep:06d}-fresh-result.json'
    if existing.exists() and json.loads(existing.read_text()).get('status')=='completed':return json.loads(existing.read_text())
    for attempt in range(2):
        try:return fresh_run(ep)
        except Exception:
            print('Fresh experiment failure',ep,'attempt',attempt+1,traceback.format_exc(),flush=True)
            r=BASE/'runs'/f'episode_{ep:06d}_newton_improvement_20261019';log=r/'logs/newton-training.log'
            if log.exists():shutil.copy2(log,r/f'research/training-failure-{time.time_ns()}.log')
    return json.loads(existing.read_text()) if existing.exists() else {'episode':ep,'status':'failed'}


def run():
    while not baseline_ready():
        status=json.loads((BATCH/'batch-status.json').read_text())
        phase('waiting_for_all_baseline_exports',baseline_exports_complete=status['rollout_exports_completed'],
              baseline_training_complete=status['policies_training_completed'])
        time.sleep(30)
    for ep in EPISODES:preserve(ep)
    subprocess.run([str(BASE/'runs/episode_000005_newton_20261015/.venv-video/bin/python'),str(BASE/'batch_failure_triage.py')],cwd=BASE,check=True)
    profiles={}
    for ep in PRIORITY:
        try:profiles[ep]=profile_trials(ep)
        except Exception as error:
            profiles[ep]={'episode':ep,'status':'failed','error':str(error),'traceback':traceback.format_exc()}
            save(OUT/f'episode_{ep:06d}-controller-result.json',profiles[ep])
    fresh=[]
    for ep in USER_PRIORITY+[36,25,8,28]:
        chosen=profiles.get(ep,{}).get('selected',baseline(ep))
        if not (chosen['single_attempt_tracking']['passed_all_frames'] and chosen['completion_rate_32_worlds']>=.5):fresh.append(ep)
    phase('fresh_training',episodes_queued=fresh,maximum_concurrent_learners=2,
          user_priority_episodes=USER_PRIORITY,
          experiment_budget_per_episode=50_000_896,
          successful_controller_trials=[ep for ep,row in profiles.items() if row.get('measured_improvement')])
    fresh_results={}
    with concurrent.futures.ThreadPoolExecutor(max_workers=2) as pool:
        pending={pool.submit(run_fresh_with_retry,ep):ep for ep in fresh}
        while pending:
            done,_=concurrent.futures.wait(pending,timeout=30,return_when=concurrent.futures.FIRST_COMPLETED)
            for future in done:
                ep=pending.pop(future)
                try:fresh_results[ep]=future.result()
                except Exception as error:fresh_results[ep]={'episode':ep,'status':'failed','error':str(error)}
            progress={}
            for ep in fresh:
                p=BASE/'runs'/f'episode_{ep:06d}_newton_improvement_20261019/flash_chord_coherent_actor_transfer/metrics.jsonl'
                if p.exists():
                    for line in reversed(p.read_text().splitlines()):
                        try:
                            value=json.loads(line);progress[str(ep)]={'additional_environment_steps':value['additional_environment_steps'],'latest_training_utc':value['utc']};break
                        except (ValueError,KeyError):pass
            phase('fresh_training',episodes_queued=fresh,
                  user_priority_episodes=USER_PRIORITY,
                  fresh_experiments_completed=[e for e,v in fresh_results.items() if v['status']=='completed'],
                  fresh_experiments_failed=[e for e,v in fresh_results.items() if v['status']=='failed'],
                  training_progress=progress)
    for ep in EPISODES:verify_preserved(ep)
    improved=sorted(set(ep for ep,row in profiles.items() if row.get('measured_improvement'))
                    |set(ep for ep,row in fresh_results.items() if row.get('measured_improvement')))
    failures=[ep for ep,row in profiles.items() if row.get('status')=='failed']+[ep for ep,row in fresh_results.items() if row.get('status')=='failed']
    recommendations={}
    for ep in EPISODES:
        chosen=baseline(ep);chosen_root=root(ep);kind='preserved_baseline'
        report=json.loads((chosen_root/'result-report.json').read_text())
        quality=report['quality_assessment'];cohort=quality['completion_rate_against_video_estimate']
        profile=profiles.get(ep,{})
        if profile.get('measured_improvement'):
            chosen=profile['selected'];kind='declared_residual_gains'
            cohort=profile['native_128_world_evaluation']['metrics']['chord_sr']
            fidelity=json.loads((chosen_root/'evaluation'/(Path(chosen['recorded_states']).stem.removesuffix('-states')+'-source-fidelity.json')).read_text())
            low={name:sum(frame['objects'][name]['iou']<.5 for frame in fidelity['frames']) for name in chosen['source_metrics']['objects']}
            quality={'all_internal_frame_checks_passed':not any(low.values()) and cohort==1. and chosen['single_attempt_tracking']['passed_all_frames'],
                     'full_human_video_match_verified':False,'source_frames_below_internal_0_5_iou':low,
                     'completion_rate_against_video_estimate':cohort,'pipeline_completion_does_not_imply_successful_imitation':True}
        fresh=fresh_results.get(ep,{})
        if fresh.get('measured_improvement') and better(fresh['candidate'],chosen) and fresh['candidate_128_world_completion']>=cohort-.0625:
            chosen=fresh['candidate'];chosen_root=Path(fresh['run']);kind='independent_fresh_training'
            quality=json.loads((chosen_root/'result-report.json').read_text())['quality_assessment']
        recommendations[str(ep)]={'run':str(chosen_root),'method':kind,'selected':chosen,'quality_assessment':quality,
                                  'recorded_states_sha256':digest(chosen_root/chosen['recorded_states']),
                                  'checkpoint_sha256':digest(chosen_root/chosen['checkpoint'].removeprefix('/episode/'))}
    save(OUT/'best-policy-sources.json',{'updated_utc':utc(),'finalized':True,'episodes':recommendations,
                                      'earlier_episodes_1_and_5_preserved':True,'original_baselines_preserved':True,
                                      'selection_rule':'Keep a variant only when source overlap, complete-attempt tracking, and 128-world completion meet the declared improvement/non-regression criteria.'})
    unresolved=[ep for ep in USER_PRIORITY if not recommendations[str(ep)]['quality_assessment']['all_internal_frame_checks_passed']]
    phase('completed_with_experiment_failures' if failures else 'completed',
          baseline_videos_complete=18,episodes_with_measured_improvements=improved,
          user_priority_episodes=USER_PRIORITY,unresolved_priority_episodes=unresolved,
          all_requested_object_motion_checks_passed=not unresolved,
          controller_results=profiles,fresh_training_results=fresh_results,failed_experiments=failures,
          official_submission_uploaded=False)


if __name__=='__main__':
    OUT.mkdir(exist_ok=True)
    with (OUT/'queue.lock').open('a') as lock:
        fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
        try:run()
        except Exception as error:
            phase('failed',error=str(error),traceback=traceback.format_exc())
            raise
