"""Compare every source frame of a fresh checkpoint against its reconstruction.

Read-only diagnostic: no policy, reference, selection, or simulator state changes.
"""
import argparse
import json
import sys
from pathlib import Path

import cv2
import numpy as np
import pyarrow.parquet as pq
import trimesh
from scipy.spatial.transform import Rotation
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

from improvement_common import save, utc


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--run', type=Path, required=True)
    args = parser.parse_args()
    root = args.run.resolve()
    cfg = json.loads((root / 'episode-config.json').read_text())
    rows = json.loads((root / 'checkpoint-reviews.json').read_text())
    row = rows[-1]
    stem = Path(row['recorded_states']).stem.removesuffix('-states')
    policy = json.loads((root / 'evaluation' / f'{stem}-source-fidelity.json').read_text())
    reference = json.loads((root / 'evaluation/reference-source-fidelity.json').read_text())
    count = cfg['frames']
    assert len(policy['frames']) == len(reference['frames']) == count
    assert [f['frame'] for f in policy['frames']] == list(range(count))
    dst = root / 'research/checkpoint-diagnosis'
    dst.mkdir(exist_ok=True)
    # The raw video fit and the physically adjusted training target differ.
    # Score both, so policy errors are not confused with reference projection.
    sys.path.insert(0, str(root / 'scripts'))
    from source_geometry import silhouette, human_occlusion, overlap
    motion = np.load(root / 'reconstruction/scene_motion_physical.npz')
    hands = np.load(root / 'reconstruction/hand_measurements.npz')
    parquet = next((root / f"human_motion_data/{row['inference_reference']}/processed").rglob('motion.parquet'))
    targets = pq.ParquetFile(parquet).read(columns=['object_body_position', 'object_body_wxyz']).to_pylist()[0]
    positions = np.asarray(targets['object_body_position'])
    quaternions = np.asarray(targets['object_body_wxyz'])
    rotations = Rotation.from_quat(quaternions.reshape(-1, 4), scalar_first=True).as_matrix().reshape(*quaternions.shape[:2], 3, 3)
    vertices = [trimesh.load(root / f'reconstruction/mesh_priors/{name}.ply', process=False).vertices for name in cfg['objects']]
    native_scores = {name: [] for name in cfg['objects']}
    for frame in range(count):
        index = int(np.rint(frame * (len(positions) - 1) / (count - 1)))
        camera = np.linalg.inv(motion['camera_to_world'][frame])
        exclude = human_occlusion(hands, frame)
        masks = [cv2.resize(cv2.imread(str(root / f'reconstruction/masks/{b+1}/{frame:06d}.png'), 0), (512, 384), interpolation=cv2.INTER_NEAREST) > 0 for b in range(len(vertices))]
        for b, name in enumerate(cfg['objects']):
            transform = np.eye(4)
            transform[:3, :3] = rotations[index, b]
            transform[:3, 3] = positions[index, b]
            rendered = silhouette(vertices[b], camera @ transform)
            valid = ~exclude.copy()
            for other, mask in enumerate(masks):
                if other != b:
                    valid &= ~mask
            native_scores[name].append(float(overlap(rendered, masks[b], valid)['iou']))
    fig, axes = plt.subplots(len(cfg['objects']), 1, figsize=(12, 3 * len(cfg['objects'])), sharex=True, squeeze=False)
    objects = {}
    for ax, name in zip(axes[:, 0], cfg['objects']):
        ref = [f['objects'][name]['iou'] for f in reference['frames']]
        actual = [f['objects'][name]['iou'] for f in policy['frames']]
        ax.plot(range(count), ref, label='Reconstruction vs human', color='#247a45')
        ax.plot(range(count), native_scores[name], label='Physics training target vs human (kinematic)', color='#315bac', linestyle='--')
        ax.plot(range(count), actual, label='Unassisted policy vs human', color='#b74b20')
        ax.axhline(.5, color='gray', linewidth=.7, linestyle=':')
        ax.set(title=name.replace('_', ' '), ylabel='Object mask IoU', ylim=(0, 1))
        ax.legend(loc='lower left', fontsize=8)
        windows = []
        for phase in policy['phases']:
            start, end = phase['start_frame'], phase['end_frame']
            indexes = range(start, end + 1)
            mean_ref = sum(ref[i] for i in indexes) / len(indexes)
            mean_policy = sum(actual[i] for i in indexes) / len(indexes)
            mean_native = sum(native_scores[name][i] for i in indexes) / len(indexes)
            windows.append({'start_frame': start, 'end_frame': end,
                            'reconstruction_mean_iou': mean_ref,
                            'policy_mean_iou': mean_policy,
                            'physics_training_target_mean_iou': mean_native,
                            'reconstruction_minus_policy_iou': mean_ref - mean_policy})
        objects[name] = {'phases': windows, 'physics_training_target_mean_iou': sum(native_scores[name]) / count, 'largest_policy_reconstruction_gap': max(windows, key=lambda x: x['reconstruction_minus_policy_iou'])}
    axes[-1, 0].set_xlabel('Original source video frame (no time warping)')
    fig.suptitle(f"Episode {cfg['episode']:06d} · checkpoint {row['environment_steps']:,}")
    fig.tight_layout()
    figure = dst / f"checkpoint-{row['environment_steps']}-source-overlap.png"
    fig.savefig(figure, dpi=150)
    plt.close(fig)
    report = {'updated_utc': utc(), 'episode': cfg['episode'], 'checkpoint_steps': row['environment_steps'],
              'all_source_frames_checked': count, 'completion_rate_32_worlds': row['completion_rate_32_worlds'],
              'objects': objects, 'figure': str(figure), 'verified_improvement': False,
              'diagnosis_version': 2,
              'limitation': 'The physics training target curve is a kinematic reference, not a policy rollout. Mask overlap is affected by human occlusion and reconstruction error. This diagnoses discrepancies; it does not verify successful imitation.'}
    save(dst / 'latest.json', report)
    print(json.dumps(report, indent=2))


if __name__ == '__main__':
    main()
