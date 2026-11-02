"""Separate new end-to-end execution, source matching, and submission packaging."""
import hashlib
import json
from pathlib import Path
from improvement_common import BASE,BATCH,save,utc
from episode_completion_gate import completion

POLICY=BATCH/'human_match_loop_20261023/full-roster-policy.json'
REPORT=BATCH/'submission_staging/full-roster-review.json'

def read(path):
    try:return json.loads(Path(path).read_text())
    except (OSError,ValueError):return {}

def manifest_version(manifest):
    try:
        identity={key:manifest[name]['sha256'] for key,name in [('comparison','original_vs_policy.mp4'),('rollout','rollout_source_timed.mp4'),('report','report.json')]}
        return hashlib.sha256(json.dumps(identity,sort_keys=True).encode()).hexdigest()[:16]
    except (KeyError,TypeError):return None

def attempt_review(root,episode,epoch,adopted):
    root=Path(root);cfg=read(root/'episode-config.json');request=read(cfg.get('match_request','/nonexistent'))
    eligible=(str(root) in adopted or bool(epoch and request.get('started_utc','')>=epoch))
    if not eligible:return {'run':None,'status':'queued','stages':{k:False for k in ['reconstruction','training','rollout','source_audit']},'pipeline_complete':False,'episode_done':False,'motion_passed':False}
    output=Path(cfg.get('match_output',str(BATCH/'nonexistent')))
    result=read(output/f'episode_{episode:06d}-fresh-result.json')
    strict=read(root/'research/strict-human-match.json')
    manifest=read(root/'deliverables/manifest.json')
    state=read(root/'research/rebuild-status.json')
    version=manifest_version(manifest)
    pinned=bool(version and strict.get('version')==version and strict.get('states_sha256')
                and strict['states_sha256']==manifest.get('policy_states.npz',{}).get('sha256'))
    stages={'reconstruction':(root/'research/full-reconstruction-completed.json').exists(),
            'training':(root/'flash_chord_coherent_actor_transfer/TRAINING_COMPLETED').exists(),
            'rollout':bool(result.get('status')=='completed' and result.get('artifact_integrity',{}).get('integrity_passed')),
            'source_audit':bool(pinned and strict.get('all_frames_checked') and strict.get('source_frames')==cfg.get('frames'))}
    complete=all(stages.values())
    return {'run':str(root),'status':'pipeline_complete' if complete else state.get('status','pending'),
            'phase':state.get('phase'),'stages':stages,'pipeline_complete':complete,
            'episode_done':bool(complete and completion(strict)['episode_done']),
            'motion_passed':bool(complete and strict.get('passes_strict_local_checks')),
            'failed_source_frames':strict.get('failed_frame_count'),'unobservable_source_frames':strict.get('unobservable_frame_count'),
            'rollout_version':version,'audit_pinned_to_recorded_states':pinned}

def build(rows,loop,policy,staging):
    expected=policy['episodes'];by_episode={row['episode']:row for row in rows};episodes={}
    active=read(BATCH/'improvements_20261021/active-runs.json')
    for ep in expected:
        row=by_episode.get(ep,{})
        run=active.get(str(ep))
        root=BASE/'runs'/run if run else BASE/'runs/nonexistent'
        attempt=attempt_review(root,ep,policy['round_epoch'],set(policy.get('adopted_runs',[])))
        check=loop.get('episodes',{}).get(str(ep),{})
        latest=row.get('videos',{}).get('latest') or {}
        matching=bool(latest and check.get('version')==latest.get('version'))
        attempt['current_selected_video']={'version':latest.get('version'),'run':latest.get('run'),
            'checked':matching,'motion_passed':bool(matching and check.get('passes_strict_local_checks')),
            'failed_source_frames':check.get('failed_frame_count') if matching else None}
        episodes[str(ep)]=attempt
    completed=[ep for ep in expected if episodes[str(ep)]['pipeline_complete']]
    matched=[ep for ep in expected if episodes[str(ep)]['motion_passed'] and episodes[str(ep)]['current_selected_video']['motion_passed']]
    done=[ep for ep in expected if episodes[str(ep)].get('episode_done') and completion(loop.get('episodes',{}).get(str(ep),{}))['episode_done']]
    blockers=[]
    if set(by_episode)!=set(expected):blockers.append('The displayed roster does not match the official20-episode roster.')
    if len(completed)!=len(expected):blockers.append('New reconstruction, training, rollout and pinned source audit are incomplete for some episodes.')
    if len(matched)!=len(expected):blockers.append('Some episodes still fail source-motion checks or contain unverified frames.')
    if len(done)!=len(expected):blockers.append('An episode is not done until exact object-motion matching is verified for every object and frame.')
    if not staging.get('final_submission_package_ready'):blockers.append('The official full-roster package and producing code commit remain unverified.')
    return {'updated_utc':utc(),'round_epoch':policy['round_epoch'],'required_episodes':expected,
        'official_roster_matches':set(by_episode)==set(expected),'new_pipelines_completed':len(completed),
        'new_pipelines_completed_episodes':completed,'motion_checks_passed':len(matched),
        'episodes_done':len(done),'episodes_done_ids':done,
        'all20_source_videos_audited':loop.get('current_round_episodes_checked')==len(expected),
        'source_frames_audited':loop.get('current_round_source_frames_checked',0),
        'ready_for_submission':not blockers,'exact_human_match_verified':False,
        'official_ground_truth_score_available':False,'blockers':blockers,'episodes':episodes}

def update(rows,loop,staging):
    policy=read(POLICY)
    if not policy:return {}
    result=build(rows,loop,policy,staging);save(REPORT,result);return result
