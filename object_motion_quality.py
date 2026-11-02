"""Object-by-object, source-time checks for the user-prioritized manipulation failures.

Consumes the complete source-fidelity sequence, without editing poses or time.
Held and transition windows prevent a stationary rack or a long pause from hiding
a dropped spoon. These are local diagnostics, not official challenge scores.
"""
from pathlib import Path
import json
import math
import statistics

PRIORITY_EPISODES = [10, 15, 16, 19, 20, 30, 32, 33]


def mean(values):
    return statistics.fmean(values) if values else None


def summarize(cfg, fidelity, tracking):
    frames = fidelity['frames']
    count = cfg['frames']
    if len(frames) != count or [f.get('frame', f.get('source_frame')) for f in frames] != list(range(count)):
        raise ValueError('Object motion quality requires every original frame, in order')
    names = cfg['objects']
    if any(set(frame['objects']) != set(names) for frame in frames):
        raise ValueError('Source/rollout object identities do not correspond')
    objects = {}
    for b, name in enumerate(names):
        grasps = [g for g in cfg['grasps'] if g['object'] == b]
        held = {i for g in grasps for i in range(g['start'], g['end'] + 1)}
        transitions = {i for g in grasps for boundary in [g['start'], g['end']]
                       for i in range(max(0, boundary - 4), min(count, boundary + 5))}
        scores = [float(frame['objects'][name]['iou']) for frame in frames]
        if not all(math.isfinite(v) and 0 <= v <= 1 for v in scores):
            raise ValueError('Invalid source overlap measurement')
        errors = [float(frame['objects'][name]['centroid_error_px']) for frame in frames
                  if frame['objects'][name]['centroid_error_px'] is not None]
        bad = [i for i, score in enumerate(scores) if score < .5]
        streak = longest = 0
        for score in scores:
            streak = streak + 1 if score < .5 else 0
            longest = max(longest, streak)
        phase_values = []
        for phase in fidelity.get('phases', []):
            # Very short boundary-only phases are already tested in transition windows.
            start, end = phase['start_frame'], phase['end_frame']
            if end - start + 1 >= 5:
                phase_values.append(mean(scores[start:end + 1]))
        objects[name] = {
            'source_frame_count': count,
            'mean_iou': mean(scores),
            'held_frame_count': len(held),
            'held_mean_iou': mean([scores[i] for i in sorted(held)]) if held else mean(scores),
            'transition_mean_iou': mean([scores[i] for i in sorted(transitions)]) if transitions else mean(scores),
            'worst_phase_mean_iou': min(phase_values) if phase_values else min(scores),
            'source_frames_below_0_5_iou': len(bad),
            'first_low_overlap_source_frame': bad[0] if bad else None,
            'longest_low_overlap_run_frames': longest,
            'mean_centroid_error_px': mean(errors),
            'native_tracking': tracking.get('per_body', {}).get(name, {}),
        }
    held_values = [v['held_mean_iou'] for v in objects.values()]
    transition_values = [v['transition_mean_iou'] for v in objects.values()]
    return {'version': 1, 'episode': cfg['episode'], 'source_frames_checked': count,
            'object_identities_match': True, 'frame_warping_used': False,
            'score': .5 * mean(held_values) + .25 * min(held_values) + .25 * mean(transition_values),
            'objects': objects,
            'every_object_passes_all_local_frames': bool(tracking.get('passed_all_frames') and
                all(v['source_frames_below_0_5_iou'] == 0 for v in objects.values())),
            'limitation': 'Source masks and approximate human occlusion have uncertainty; local checks do not verify exact human imitation.'}


def attach(root, row):
    root = Path(root)
    if not (root / 'episode-config.json').exists():
        return row
    cfg = json.loads((root / 'episode-config.json').read_text())
    if cfg['episode'] not in PRIORITY_EPISODES or not row['single_attempt_tracking'].get('all_frames_finite', True):
        return row
    if row.get('object_motion_quality', {}).get('version') == 1:
        return row
    stem = Path(row['recorded_states']).stem.removesuffix('-states')
    fidelity = json.loads((root / 'evaluation' / f'{stem}-source-fidelity.json').read_text())
    return {**row, 'object_motion_quality': summarize(cfg, fidelity, row['single_attempt_tracking'])}


def nonregressing_objects(candidate, original):
    """Require correspondence for both objects, and for held/release phases."""
    new, old = candidate.get('object_motion_quality'), original.get('object_motion_quality')
    if new is None and old is None:
        return True
    if new is None or old is None or not new['object_identities_match'] or set(new['objects']) != set(old['objects']):
        return False
    for name, before in old['objects'].items():
        after = new['objects'][name]
        for metric, allowance in [('mean_iou', .01), ('held_mean_iou', .02),
                                  ('transition_mean_iou', .03), ('worst_phase_mean_iou', .03)]:
            if after[metric] < before[metric] - allowance:
                return False
        # No material 3D regression can be hidden by a favorable camera projection.
        for metric, allowance in [('mean_position_error_m', .005), ('mean_orientation_error_rad', .05)]:
            a, b = after['native_tracking'].get(metric), before['native_tracking'].get(metric)
            if a is not None and b is not None and a > b + allowance:
                return False
    return new['score'] >= old['score'] + .005


def ranking_score(row):
    return row.get('object_motion_quality', {}).get('score', row['source_agreement_score'])
