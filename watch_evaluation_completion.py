"""Finalize the current round, audit all20 videos and stage the accepted policies."""
import json,subprocess,time
from improvement_common import BASE,BATCH,save,utc

OUT=BATCH/'improvements_20261021'
ROOTS={30:BASE/'runs/episode_000030_newton_observed_brush_20261021',
       32:BASE/'runs/episode_000032_newton_contact_precision_full_20261021'}


def main():
    while True:
        registry=OUT/'active-runs.json'
        if registry.exists():ROOTS.update({int(ep):BASE/'runs'/name for ep,name in json.loads(registry.read_text()).items()})
        rows={}
        for ep,root in ROOTS.items():
            path=root/'research/improvement-status.json';row=json.loads(path.read_text()) if path.exists() else {'status':'reconstructing'}
            metrics=root/'flash_chord_coherent_actor_transfer/metrics.jsonl'
            if metrics.exists() and metrics.stat().st_size:
                last=json.loads(metrics.read_text().splitlines()[-1]);row={**row,'additional_environment_steps':last['additional_environment_steps'],'last_metric_utc':last['utc']}
            rows[str(ep)]={k:v for k,v in row.items() if k not in ['baseline','candidate','artifact_integrity','traceback']}
        completed=[ep for ep in ROOTS if rows[str(ep)]['status']=='completed']
        failed=[ep for ep in ROOTS if rows[str(ep)]['status']=='failed']
        phase='completed' if len(completed)==2 else 'needs_attention' if failed else 'contact_refinement_and_precision_training'
        save(OUT/'phase-status.json',{'phase':phase,'updated_utc':utc(),'episodes':[30,32],
             'completed':completed,'failed':failed,'running_or_queued':[ep for ep in ROOTS if ep not in completed+failed],
             'episodes_status':rows,'evaluation_episode_count':20,'full_human_motion_match_verified':False})
        if len(completed)==2:break
        time.sleep(20)
    log=OUT/'all20-finalization.log'
    with log.open('a') as stream:
        for executable,args in [(BASE/'runs/episode_000005_newton_20261015/.venv-video/bin/python',['evaluation_inventory.py']),
                                (BASE/'.venv/bin/python',['policy_recommendations.py','--finalize']),
                                (BASE/'.venv-submission/bin/python',['batch_submission_stage.py'])]:
            subprocess.run([str(executable),'-u',*args],cwd=BASE,stdout=stream,stderr=subprocess.STDOUT,check=True)
    staged=json.loads((BATCH/'submission_staging/readiness.json').read_text());assert staged['all_20_episode_data_staged']
    audit=json.loads((BATCH/'all20_evaluation/all20-episode-audit.json').read_text());assert audit['all_20_pipeline_deliverables_complete_and_verified']
    save(OUT/'all20-final-completion.json',{'updated_utc':utc(),'all_20_episode_deliverables_verified':True,
         'all_20_accepted_policy_trajectories_staged':True,'audit':'all20_evaluation/all20-episode-audit.json',
         'newest_verified_video_links_use_accepted_policies':True,'official_submission_uploaded':False,
         'exact_human_video_match_verified':False})
    print('All20 completed, accepted videos audited, and selected trajectories staged.',flush=True)


if __name__=='__main__':main()
