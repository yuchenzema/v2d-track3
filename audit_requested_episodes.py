"""Recheck saved episode20/30 evidence without launching or certifying a rollout."""
import json
from pathlib import Path

import numpy as np

from assess_episode30_pickup import assess_file
from human_match_criteria import LIMITS, assess
from improvement_common import BASE, BATCH, digest, save, utc

OUT = BATCH / 'requested_episodes_20_30'


def audit_saved_report(episode, directory):
    report_path = directory / 'report.json'
    frames_path = directory / 'frame-metrics.json'
    report = json.loads(report_path.read_text())
    data = json.loads(frames_path.read_text())
    root = Path(report['run'])
    cfg = json.loads((root / 'episode-config.json').read_text())
    assert cfg['episode'] == report['episode'] == episode, 'Episode identity mismatch'
    assert cfg['frames'] == report['source_frames'], 'Source frame count mismatch'
    assert set(report['objects']) == set(cfg['objects']), 'Object identities mismatch'
    for frame in data['frames']:
        assert set(frame['objects']) == set(cfg['objects']), 'Object missing from frame'
        assert set(frame['hands']) == {'left', 'right'}, 'Hand evidence missing'
        assert 'time_s' in frame and 'recorded_control_frame' in frame, 'Source clock missing'
        for obj in frame['objects'].values():
            for key in ['rollout_orientation_world_xyzw', 'reconstruction_orientation_world_xyzw']:
                orientation = np.asarray(obj[key])
                assert orientation.shape == (4,) and np.isfinite(orientation).all(), 'Invalid orientation'
    assert digest(cfg['source']) == report['source_video_sha256'], 'Source video changed'
    selected = json.loads((root / 'final-controller-selection.json').read_text())['selected']
    recorded = Path(selected['recorded_states'])
    if str(recorded).startswith('/episode/'):
        recorded = root / str(recorded).removeprefix('/episode/')
    elif not recorded.is_absolute():
        recorded = root / recorded
    assert digest(recorded) == report['recorded_states_sha256'], 'Recorded rollout changed'
    with np.load(recorded) as states:
        clock = states['reference_frames']
        expected_count = 2 * (cfg['frames'] - 1)
        physical = {
            'continuous_source_clock': bool(np.array_equal(clock, np.arange(expected_count))),
            'all_physical_states_finite': bool(len(states['body_poses']) == expected_count
                and np.isfinite(states['body_poses']).all()),
            'zero_object_assistance': bool(states['object_assistance'].size >= expected_count
                and len(states['object_assistance']) == expected_count
                and np.all(states['object_assistance'] == 0)),
        }
    strict = assess(report, data)
    hands = {}
    for side in ['left', 'right']:
        failures, unobserved = [], []
        for frame in data['frames']:
            hand = frame['hands'][side]
            if not hand['observed']:
                unobserved.append(frame['frame'])
                continue
            errors = [hand.get(k) for k in ['wrist_error_px', 'mean_joint_error_px', 'fingertip_error_px']]
            if any(v is None or not np.isfinite(v) or v > LIMITS['maximum_centroid_error_px'] for v in errors):
                failures.append(frame['frame'])
        hands[side] = {'failed_frames': failures, 'unobserved_frames': unobserved,
                       'maximum_projection_error_px': LIMITS['maximum_centroid_error_px']}
    result = {
        'episode': episode, 'run': str(root), 'updated_utc': utc(),
        'evidence_kind': 'Reassessment of an existing saved rollout; no new simulation performed',
        'source_video_sha256': report['source_video_sha256'],
        'recorded_states_sha256': report['recorded_states_sha256'],
        'frame_metrics_sha256': digest(frames_path),
        'strict_objects': strict, 'hands': hands, 'physical_checks': physical,
        'all_component_local_checks_pass': bool(strict['passes_strict_local_checks']
            and all(physical.values())
            and all(not h['failed_frames'] and not h['unobserved_frames'] for h in hands.values())),
        'exact_human_match_verified': False,
        'limitation': 'Local tolerances and inferred 3D references cannot establish exact equivalence; occluded components remain unverified.',
    }
    save(OUT / f'episode_{episode:06d}-saved-audit.json', result)
    return result


def main():
    results = {}
    for episode in [20, 30]:
        directory = BATCH / f'improvements_20261023_rebuild/human_review/episode_{episode:06d}'
        results[episode] = audit_saved_report(episode, directory)

    pickup_run = BASE / 'runs/episode_000030_newton_pickup_repair_20261026'
    latest = json.loads((pickup_run / 'checkpoint-reviews.json').read_text())[-1]
    states = pickup_run / latest['recorded_states']
    pickup = assess_file(pickup_run, states)
    pickup.update(updated_utc=utc(), states=str(states), states_sha256=digest(states),
                  environment_steps=latest['environment_steps'], new_rollout_performed=False)
    save(OUT / 'episode_000030-latest-saved-pickup.json', pickup)

    path = OUT / 'status.json'
    status = json.loads(path.read_text()) if path.exists() else {
        'requested_episodes': [20, 30],
        'requested_stages': ['rebuild', 'training', 'rollout', 'all_component_source_comparison'],
        'requested_utc': utc(), 'status': 'blocked', 'active': False,
        'reason': 'This session denies Docker, GPU devices, user-service access, and local listening sockets.',
        'blockers': ['host_execution_access', 'SAM3D_model_access_and_metric_alignment'],
        'new_reconstruction_started': False, 'new_training_started': False,
        'new_rollout_started': False,
        'acceptance': {'every_original_frame': True, 'all_named_objects_and_both_hands': True,
                       'source_timing_preserved': True, 'zero_object_assistance': True,
                       'physical_pickup_and_sustained_holding': True,
                       'release_and_final_placement': True, 'occluded_components_verified': False},
    }
    status.update(updated_utc=utc(), offline_audit_completed_utc=utc(), exact_human_match_verified=False)
    status['episodes'] = {
        str(ep): {
            'episode': ep,
            'objects': list(row['strict_objects']['objects']),
            'source_frames_checked': row['strict_objects']['source_frames'],
            'failed_source_frames': row['strict_objects']['failed_frame_count'],
            'reconstruction_mismatch_frames': row['strict_objects']['reconstruction_mismatch_frame_count'],
            'all_component_local_checks_pass': row['all_component_local_checks_pass'],
            'audit': str(OUT / f'episode_{ep:06d}-saved-audit.json'),
            'stages': status.get('episodes', {}).get(str(ep), {}).get('stages',
                {name: 'not_started' for name in ['rebuild', 'training', 'rollout']}),
        } for ep, row in results.items()
    }
    status['latest_saved_brush_pickup'] = {
        'steps': latest['environment_steps'], 'passes': pickup['passes'],
        'report': str(OUT / 'episode_000030-latest-saved-pickup.json'),
    }
    save(path, status)
    print(json.dumps(status, indent=2))


if __name__ == '__main__':
    main()
