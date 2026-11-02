"""Rank baseline tracking failures and inspect their original source frames on CPU."""
from pathlib import Path
import hashlib
import importlib.util
import json
import time
import cv2
import numpy as np
import trimesh
from batch_artifacts import complete

BASE = Path(__file__).resolve().parent
BATCH = BASE / 'runs/evaluation_batch_20261015'
OUT = BATCH / 'improvements_20261019'


def triage():
    OUT.mkdir(exist_ok=True)
    rows = []
    for root in sorted((BASE / 'runs').glob('episode_*_newton_batch_20261015')):
        if not complete(root):
            continue
        cfg = json.loads((root / 'episode-config.json').read_text())
        selected = json.loads((root / 'final-controller-selection.json').read_text())['selected']
        tracking = selected['single_attempt_tracking']
        report = json.loads((root / 'result-report.json').read_text())
        T = cfg['frames']
        first = tracking.get('first_violation_control_frame')
        N = tracking['recorded_control_frames']
        source_first = int(round(first * (T - 1) / (N - 1))) if first is not None else None
        state_path = root / selected['recorded_states']
        fidelity_path = root / 'evaluation' / (state_path.stem.removesuffix('-states') + '-source-fidelity.json')
        fidelity = json.loads(fidelity_path.read_text())
        mean_iou = np.array([np.mean([v['iou'] for v in f['objects'].values()])
                             for f in fidelity['frames']])
        worst = int(np.argmin(np.convolve(np.pad(mean_iou, (2, 2), mode='edge'), np.ones(5)/5, mode='valid')))
        fractions = tracking.get('violation_frame_count', 0) / N
        row = {
            'episode': cfg['episode'], 'source_frames': T, 'control_frames': N,
            'selected_checkpoint': selected['checkpoint'],
            'source_mean_object_iou': selected['source_mean_object_iou'],
            'native_128_world_completion': report['quality_assessment']['completion_rate_against_video_estimate'],
            'selected_attempt_tracks_every_frame': tracking['passed_all_frames'],
            'violating_frame_fraction': fractions,
            'first_violation_source_frame': source_first,
            'first_violation_source_time_s': source_first / 20 if source_first is not None else None,
            'failure_timing': 'none' if first is None else 'early' if source_first < .35 * T else 'late' if source_first > .65 * T else 'middle',
            'per_object_errors': tracking['per_body'],
            'maximum_wrist_error_m': tracking['max_wrist_position_error_m'],
            'worst_source_overlap_frame': worst,
            'observed_grasps': cfg['grasps'],
            'source_notes': cfg.get('source_phase_notes', cfg.get('source_review_notes', '')),
            'priority_score': float(2 * fractions + 1 - selected['source_mean_object_iou']),
            'original_result_report': str(root / 'result-report.json'),
        }
        # New diagnostics never alter a baseline policy, reference, or video.
        signature = hashlib.sha256((root / 'deliverables/manifest.json').read_bytes()).hexdigest()
        marker = OUT / f'episode_{cfg["episode"]:06d}-triage.json'
        previous = json.loads(marker.read_text()) if marker.exists() else {}
        image_path = OUT / f'episode_{cfg["episode"]:06d}-failure-review.jpg'
        if previous.get('baseline_manifest_sha256') != signature or not image_path.exists():
            spec = importlib.util.spec_from_file_location(f'source_geometry_{cfg["episode"]}', root / 'scripts/source_geometry.py')
            geometry = importlib.util.module_from_spec(spec)
            spec.loader.exec_module(geometry)
            motion = np.load(root / 'reconstruction/scene_motion_physical.npz')
            states = np.load(state_path)
            names = states['body_names'].tolist()
            meshes = [trimesh.load(root / f'reconstruction/mesh_priors/{name}.ply', process=False)
                      for name in cfg['objects']]
            requested = [0, source_first or 0, min(T - 1, (source_first or 0) + 5), worst, int(.8 * T), T - 1]
            cap = cv2.VideoCapture(str(root / 'reconstruction/calibrated.mp4'))
            panels = []
            for frame in requested:
                cap.set(cv2.CAP_PROP_POS_FRAMES, frame)
                ok, image = cap.read()
                if not ok:
                    raise ValueError(f'Cannot decode original calibrated frame {frame} of {root.name}')
                image = cv2.resize(image, (512, 384))
                C = np.linalg.inv(motion['camera_to_world'][frame])
                actual = states['body_poses'][int(np.rint(frame * (N - 1) / (T - 1)))]
                for b, name in enumerate(cfg['objects']):
                    source = cv2.imread(str(root / f'reconstruction/masks/{b+1}/{frame:06d}.png'), 0)
                    source = cv2.resize(source, (512, 384), interpolation=cv2.INTER_NEAREST)
                    contours, _ = cv2.findContours(source, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
                    cv2.drawContours(image, contours, -1, (70, 230, 70), 1)
                    rendered = geometry.silhouette(meshes[b].vertices,
                                                  C @ geometry.matrix(actual[names.index(f'track3_{name}/object')]))
                    contours, _ = cv2.findContours(rendered.astype(np.uint8)*255, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
                    cv2.drawContours(image, contours, -1, (230, 60, 230), 2)
                cv2.rectangle(image, (0, 0), (512, 49), (20, 20, 20), -1)
                cv2.putText(image, f'Episode {cfg["episode"]} | source frame {frame} | overlap {mean_iou[frame]:.2f}',
                            (8, 20), 0, .48, (255, 255, 255), 1, cv2.LINE_AA)
                cv2.putText(image, 'green: video mask | magenta: actual learned-policy pose',
                            (8, 39), 0, .42, (255, 255, 255), 1, cv2.LINE_AA)
                panels.append(image)
            cap.release()
            cv2.imwrite(str(image_path), np.concatenate([np.concatenate(panels[:3], axis=1),
                                                       np.concatenate(panels[3:], axis=1)]))
        row.update(baseline_manifest_sha256=signature, failure_review_image=str(image_path))
        marker.write_text(json.dumps(row, indent=2))
        rows.append(row)
    rows.sort(key=lambda row: row['priority_score'], reverse=True)
    result = {'updated_utc': time.strftime('%Y-%m-%dT%H:%M:%SZ', time.gmtime()),
              'baseline_episodes_available': len(rows),
              'method': __doc__, 'official_held_out_scores': False,
              'causality_note': 'Failure timing and projection errors identify where to inspect; they do not alone prove a reconstruction or controller cause.',
              'episodes': rows}
    (OUT / 'failure-triage.json').write_text(json.dumps(result, indent=2))
    print('CPU failure triage complete:', len(rows), 'episodes; priority', [r['episode'] for r in rows], flush=True)
    return result


if __name__ == '__main__':
    triage()
