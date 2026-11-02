"""Keep final export recommendations consistent with the verified GUI selection.

Later optimization rounds must not be lost when the original queue finalizes its
older result roster. Preview is read-only apart from its separate report file;
final recommendations are written only after every registered round completes.
"""
from pathlib import Path
import argparse
import hashlib
import json
import time
from improvement_common import BATCH,OUT,EPISODES,root,digest,save,utc
from video_catalog import catalog

ROUNDS=['improvements_20261019','improvements_20261020','improvements_20261020_grasp','improvements_20261020_backward','improvements_20261020_contacts']
TERMINAL={'completed','completed_with_experiment_failures'}


def rounds_finished():
    phases={}
    rounds=ROUNDS+(['improvements_20261021'] if (BATCH/'improvements_20261021/phase-status.json').exists() else [])
    if (BATCH/'improvements_20261022_rebuild/phase-status.json').exists():rounds.append('improvements_20261022_rebuild')
    if (BATCH/'improvements_20261022_policy_repeat19/phase-status.json').exists():rounds.append('improvements_20261022_policy_repeat19')
    if (BATCH/'improvements_20261023_rebuild/phase-status.json').exists():rounds.append('improvements_20261023_rebuild')
    for name in rounds:
        path=BATCH/name/'phase-status.json'
        phases[name]=json.loads(path.read_text()).get('phase') if path.exists() else None
    loop=BATCH/'human_match_loop_20261023/status.json'
    if loop.exists():phases['human_match_loop_20261023']=json.loads(loop.read_text()).get('status')
    return all(phase in TERMINAL for phase in phases.values()),phases


def snapshot(finalize=False):
    finished,phases=rounds_finished()
    if finalize and not finished:raise ValueError('Optimization rounds are still running')
    episodes={}
    for ep in EPISODES:
        releases=catalog(ep,root(ep),include_selected=True)
        if releases['issues']:raise ValueError(f'Episode {ep} release integrity: {releases["issues"]}')
        release=releases['best']
        if release is None:raise ValueError(f'Episode {ep} has no verified release')
        selected=release['selected'];run=Path(release['run'])
        checkpoint=run/selected['checkpoint'].removeprefix('/episode/')
        checkpoint_hash=digest(checkpoint)
        if selected.get('checkpoint_sha256') and checkpoint_hash!=selected['checkpoint_sha256']:
            raise ValueError(f'Episode {ep} selected actor checksum mismatch')
        state=run/selected['recorded_states'];state_hash=digest(state)
        if release['kind'] in ['baseline','experiment']:
            manifest=json.loads((run/'deliverables/manifest.json').read_text())
            if checkpoint_hash!=manifest['policy.safetensors']['sha256'] or state_hash!=manifest['policy_states.npz']['sha256']:
                raise ValueError(f'Episode {ep} selected physics/checkpoint differ from the validated video release')
        combination=selected.get('actor_combination')
        if combination:
            alternate=run/combination['alternate_checkpoint'].removeprefix('/episode/')
            if digest(alternate)!=combination['alternate_checkpoint_sha256']:
                raise ValueError(f'Episode {ep} alternate actor checksum mismatch')
        method=('preserved_baseline' if release['kind']=='baseline' else
                'robot_wrist_tracking_feedback' if selected.get('robot_tracking_feedback_profile') else
                'fixed_hand_actor_combination' if combination else
                'declared_grip_correction' if selected.get('grip_bias_profile') else
                'declared_residual_gains' if release['kind']=='controller' else
                'backward_grasp_curriculum' if '_backward_grasp_' in run.name else
                'opening_grasp_curriculum' if '_grasp_curriculum_' in run.name else
                'independent_fresh_training')
        controls={k:selected.get(k) for k in ['wrist_reference_velocity_gain','physics_substeps',
            'inference_reference','robot_release_controller_spec','residual_gain_profile','actor_combination','grip_bias_profile','robot_tracking_feedback_profile',
            'robot_wrist_calibration_profile','brush_bristle_compliance_profile']}
        episodes[str(ep)]={'run':str(run),'method':method,'selected':selected,
            'release_kind':release['kind'],
            'quality_assessment':release['quality'],'recorded_states_sha256':state_hash,
            'checkpoint_sha256':checkpoint_hash,'video_release_version':release['version'],
            'reproduction_controls':controls,
            'reproduction_controls_sha256':hashlib.sha256(json.dumps(controls,sort_keys=True).encode()).hexdigest()}
    result={'updated_utc':utc(),'finalized':finalize,'optimization_rounds':phases,
        'episodes':episodes,'earlier_episodes_1_and_5_preserved':True,'original_baselines_preserved':True,
        'selection_rule':'Same verified best selection as the GUI, including later rounds; per-object source agreement, continuous tracking, and 128-world completion must meet non-regression checks.',
        'all_requested_object_motion_checks_passed':all(v['quality_assessment'].get('all_internal_frame_checks_passed',False) for v in episodes.values()),
        'official_submission_uploaded':False}
    path=OUT/'best-policy-sources.json' if finalize else BATCH/'improvements_20261020_grasp/best-policy-sources-preview.json'
    save(path,result)
    print(json.dumps({'updated_utc':result['updated_utc'],'finalized':finalize,
        'improved_episodes':[int(ep) for ep,row in episodes.items() if row['method']!='preserved_baseline']}),flush=True)
    return result


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--watch',action='store_true')
    parser.add_argument('--finalize',action='store_true')
    args=parser.parse_args()
    if args.watch:
        while not rounds_finished()[0]:time.sleep(30)
    snapshot(finalize=args.watch or args.finalize)
