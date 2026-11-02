"""Inspect reference hand/object gaps in the native collision model, without stepping.

Run with run_geometry_cpu.py. Outputs are diagnostics; no reference, checkpoint,
policy state or collision geometry is changed.
"""
import json
import os
from pathlib import Path
import runpy
import sys

import mujoco
import numpy as np
import flash_chord.training.flash_sac.runner as runner

sys.path.insert(0, '/episode/scripts')
from grasp_physics_checks import gravity_cone_feasible, native_body_id
from native_pinch_contacts import valid_distance_witness

ROOT = Path('/episode')
CONFIG = json.loads((ROOT / 'episode-config.json').read_text())
REFERENCE = CONFIG['training_reference']


def inspect(env, *args, **kwargs):
    e = env.env
    model, data = e.solver.mj_model, e.solver.mj_data
    model.opt.disableflags &= ~int(mujoco.mjtDisableBit.mjDSBL_NATIVECCD)
    active = (model.geom_contype != 0) | (model.geom_conaffinity != 0)
    labels = [mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_BODY, i) or ''
              for i in range(model.nbody)]
    rows = []
    for grasp in CONFIG['grasps']:
        name = CONFIG['objects'][grasp['object']]
        body = native_body_id(lambda label: mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, label),
                              f'track3_{name}/object')
        object_geoms = np.flatnonzero(active & (model.geom_bodyid == body))
        frames = sorted(set([grasp['start'], min(grasp['start'] + 12, grasp['end']),
                             (grasp['start'] + grasp['end']) // 2, grasp['end']]))
        for frame in frames:
            native_frame = round(frame * (e.reference.num_frames - 1) / (CONFIG['frames'] - 1))
            e.reset(frame_id=native_frame)
            e.solver._update_mjc_data(data, e.model, e.state_0)
            mujoco.mj_forward(model, data)
            contacts = []
            for finger in ['thumb', 'index', 'middle', 'ring', 'pinky']:
                finger_bodies = [i for i, label in enumerate(labels)
                                 if label.startswith(f"{grasp['side']}_sharpa_wave_")
                                 and f'_{finger}_' in label]
                hand_geoms = np.flatnonzero(active & np.isin(model.geom_bodyid, finger_bodies))
                best = None
                for hand in hand_geoms:
                    for obj in object_geoms:
                        if np.linalg.norm(data.geom_xpos[hand] - data.geom_xpos[obj]) > (
                                model.geom_rbound[hand] + model.geom_rbound[obj] + .04):
                            continue
                        points = np.zeros(6)
                        distance = float(mujoco.mj_geomDistance(model, data, int(hand), int(obj), .04, points))
                        if not valid_distance_witness(distance, points, .04) or best is not None and distance >= best['distance_m']:
                            continue
                        delta = points[3:] - points[:3]
                        normal = delta / distance if abs(distance) > 1e-9 else np.zeros(3)
                        best = dict(finger=finger, distance_m=distance, hand_geom=int(hand),
                                    object_geom=int(obj), hand_body=labels[int(model.geom_bodyid[hand])],
                                    position=points[3:].tolist(), normal_on_object=normal.tolist(),
                                    friction=float(min(model.geom_friction[hand, 0], model.geom_friction[obj, 0])))
                contacts.append(best or dict(finger=finger, distance_m=None))
            near = [c for c in contacts if c['distance_m'] is not None and abs(c['distance_m']) <= .001
                    and np.linalg.norm(c['normal_on_object']) > .5]
            support = gravity_cone_feasible([c['normal_on_object'] for c in near],
                                            [c['position'] for c in near], [c['friction'] for c in near],
                                            model.body_mass[body], data.xipos[body], gravity=model.opt.gravity)
            rows.append(dict(source_frame=frame, native_frame=native_frame, object=name, side=grasp['side'],
                             finger_contacts=contacts, near_contact_count=len(near),
                             reference_gravity_cone_feasible=support))
    report = dict(diagnostic_only=True, policy_inference=False, physics_steps_executed=0,
                  reference_modified=False, rows=rows)
    output = Path(os.environ.get('V2D_GRASP_DIAGNOSTIC_OUTPUT', '/tmp/reference-grasp-contacts.json'))
    output.write_text(json.dumps(report, indent=2))
    print(json.dumps(report), flush=True)
    raise SystemExit(0)


runner.run_training = inspect
sys.argv = [sys.argv[0], 'experiment=sharpa_flash_sac',
            f"task.parquet='/episode/human_motion_data/{REFERENCE}/processed/sequence_id={CONFIG['sequence']}/robot_name=sharpa_wave'",
            'task.motion_speed=1.0', 'reset.reset_finger_openness=0.0', 'scene.world_count=1',
            'logging.mode=disabled', 'output_dir=/tmp/grasp-contact-diagnostic',
            'hydra.run.dir=/tmp/grasp-contact-diagnostic/hydra']
runpy.run_path('/workspace/scripts/train_flash_sac.py', run_name='__main__')
