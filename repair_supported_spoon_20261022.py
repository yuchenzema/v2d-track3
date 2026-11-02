"""Refit supported spoon poses rigidly on the observed table plane.

The old landmark tracker scaled camera rays when an occluded handle appeared
shorter. That moved a supported rigid tool below the table. Here the mesh keeps
its metric dimensions and only planar translation/yaw are fitted to source pixels.
"""
import argparse
import importlib.util
import json
import shutil
from pathlib import Path

import cv2
import numpy as np
import trimesh
from scipy.ndimage import distance_transform_edt, map_coordinates
from scipy.optimize import least_squares

from improvement_common import save, utc

p = argparse.ArgumentParser(description=__doc__)
p.add_argument('--run', type=Path, required=True)
a = p.parse_args()
r = a.run.resolve()
c = json.loads((r / 'episode-config.json').read_text())
assert c['episode'] == 33
assert not (r / 'coherent-input-checksums.json').exists()
rec = r / 'reconstruction'
marker = r / 'research/supported-spoon-repair.json'
if marker.exists():
    raise SystemExit(0)
spec = importlib.util.spec_from_file_location('spoon_source_geometry', r / 'scripts/source_geometry.py')
geo = importlib.util.module_from_spec(spec)
spec.loader.exec_module(geo)
m = np.load(rec / 'scene_motion_physical.npz')
out = {k: m[k].copy() for k in m.files}
hands = np.load(rec / 'hand_measurements.npz')
tracking = json.loads((rec / 'spoon-geometric-tracking.json').read_text())
landmarks = np.asarray(tracking['canonical_head_handle_landmarks_m'])
vertices = trimesh.load(rec / 'mesh_priors/wooden_spoon.ply', process=False).vertices
sample = vertices[np.linspace(0, len(vertices)-1, min(1800, len(vertices))).astype(int)]
axis = landmarks[1] - landmarks[0]
axis /= np.linalg.norm(axis)
normal = np.array([1., 0., 0.])
normal -= axis * (normal @ axis)
normal /= np.linalg.norm(normal)
basis = np.column_stack([normal, np.cross(axis, normal), axis])
intervals = [[0, 56], [289, c['frames'] - 1]]
records = []
for frame in range(c['frames']):
    if not any(first <= frame <= last for first, last in intervals):
        continue
    world_from_camera = out['camera_to_world'][frame]
    camera_from_world = np.linalg.inv(world_from_camera)
    measured = tracking['frames'][frame]
    head, end = np.asarray(measured['head_uv']), np.asarray(measured['handle_end_uv'])
    rays = np.column_stack([[head, end], np.ones(2)]) @ np.linalg.inv(geo.K).T
    direction = rays @ world_from_camera[:3, :3].T
    # The landmark lies on the upper face of a 5mm-thick spoon.
    upper_z = .6505 + float(np.ptp(vertices[:, 0]))
    points = world_from_camera[:3, 3] + direction * ((upper_z-world_from_camera[2, 3])/direction[:, 2])[:, None]
    world_axis = points[1] - points[0]
    yaw = float(np.arctan2(world_axis[1], world_axis[0]))
    mask = cv2.resize(cv2.imread(str(rec / f'masks/2/{frame:06d}.png'), 0), (512, 384), interpolation=cv2.INTER_NEAREST) > 0
    valid = ~geo.human_occlusion(hands, frame)
    valid &= cv2.resize(cv2.imread(str(rec / f'masks/1/{frame:06d}.png'), 0), (512, 384), interpolation=cv2.INTER_NEAREST) == 0
    distance = distance_transform_edt(~mask)

    def pose(x):
        along = np.array([np.cos(x[2]), np.sin(x[2]), 0.])
        up = np.array([0., 0., 1.])
        rotation = np.column_stack([up, np.cross(along, up), along]) @ basis.T
        translation = points[0] - rotation @ landmarks[0]
        translation[:2] += x[:2]
        translation[2] = .6505 - (vertices @ rotation.T)[:, 2].min()
        matrix = np.eye(4)
        matrix[:3, :3], matrix[:3, 3] = rotation, translation
        return matrix

    def residual(x):
        matrix = camera_from_world @ pose(x)
        camera_points = sample @ matrix[:3, :3].T + matrix[:3, 3]
        uv = geo.project(camera_points) * .5
        clipped = np.clip(uv, [0, 0], [511, 383])
        visibility = map_coordinates(valid.astype(float), clipped[:, ::-1].T, order=0, mode='nearest')
        sdf = map_coordinates(distance, clipped[:, ::-1].T, order=1, mode='nearest') * visibility
        head_uv = geo.project((matrix[:3, :3] @ landmarks[0] + matrix[:3, 3])[None])[0]
        return np.concatenate([sdf / np.sqrt(len(sdf)) / 2, (head_uv-head)/6,
                               np.asarray(x[:2])/.05, [(x[2]-yaw)/.5]])

    fit = least_squares(residual, [0, 0, yaw], bounds=([-.04, -.04, yaw-.35], [.04, .04, yaw+.35]), loss='soft_l1', max_nfev=60)
    before = out['object_to_world'][frame, 1].copy()
    after = pose(fit.x)
    score_before = geo.overlap(geo.silhouette(vertices, camera_from_world @ before), mask, valid)['iou']
    score_after = geo.overlap(geo.silhouette(vertices, camera_from_world @ after), mask, valid)['iou']
    minimum_z = float((vertices @ after[:3, :3].T + after[:3, 3])[:, 2].min())
    assert abs(minimum_z-.6505) < 1e-6
    out['object_to_world'][frame, 1] = after
    records.append({'frame': frame, 'minimum_mesh_z_m': minimum_z, 'source_iou_before': score_before,
                    'source_iou_after': score_after, 'old_landmark_ray_scale': measured['rigid_length_depth_scale'],
                    'translation_correction_m': float(np.linalg.norm(after[:3, 3]-before[:3, 3]))})
shutil.copy2(rec / 'scene_motion_physical.npz', rec / 'scene_motion_before_supported_spoon_repair.npz')
np.savez_compressed(rec / 'scene_motion_physical.npz', **out)
c['source_support_intervals']['wooden_spoon'] = intervals
save(r / 'episode-config.json', c)
save(marker, {'updated_utc': utc(), 'method': __doc__, 'supported_source_intervals': intervals,
              'support_evidence': 'Every source frame reviewed; spoon remains on table until lift starts around57-58 and returns around289. Hand release remains333.',
              'mesh_scale_or_mass_changed': False, 'original_source_time_grid_preserved': True,
              'records': records, 'mean_source_iou_before': float(np.mean([x['source_iou_before'] for x in records])),
              'mean_source_iou_after': float(np.mean([x['source_iou_after'] for x in records]))})
print('Rigid table-supported spoon fit', len(records), 'frames; mean source IoU', np.mean([x['source_iou_before'] for x in records]), '->', np.mean([x['source_iou_after'] for x in records]), flush=True)
