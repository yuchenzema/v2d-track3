"""CPU-only replay of collision solids at every recorded native policy frame."""
import hashlib
import json
import os
from pathlib import Path
import runpy
import sys
import mujoco
import numpy as np
import flash_chord.training.flash_sac.runner as runner
from rigid_pose_collision_audit import PoseCollisionAuditor, NUMERICAL_EPSILON_M
from episode_clock import control_frames,motion_speed

ROOT = Path('/episode')
cfg = json.loads((ROOT / 'episode-config.json').read_text())
if cfg.get('validated_video_pipeline'):
    from legacy_source_frame_playback import install
    install()
states_path = Path(os.environ['V2D_COLLISION_STATES'])
output = Path(os.environ['V2D_COLLISION_OUTPUT'])
reference = os.environ.get('V2D_COLLISION_REFERENCE', cfg.get('training_reference', 'track3_coherent_variant'))


def inspect(env, *args, **kwargs):
    e = env.env; solver = e.solver; model = solver.mj_model; data = solver.mj_data
    assert e.model.world_count == 1
    state = np.load(states_path)
    poses = state['body_poses']; names = state['body_names'].tolist()
    assert len(poses) == control_frames(cfg)
    assert np.array_equal(state['reference_frames'], np.arange(len(poses)))
    assert np.isfinite(poses).all() and np.all(state['object_assistance'] == 0)
    assert names == list(e.model.body_label), 'Recorded states and native geometry have different body order'
    # Use native FK once to recover each collision solid's exact local transform.
    # No integration, policy inference, pose correction, or object force is applied.
    e.reset(frame_id=0); solver._update_mjc_data(data, e.model, e.state_0); mujoco.mj_forward(model, data)
    template = e.state_0.body_q.numpy()
    auditor = PoseCollisionAuditor(model, data, names, template, cfg['objects'])
    before_pos, before_mat = data.geom_xpos.copy(), data.geom_xmat.copy()
    auditor.inspect(template)
    assert np.max(np.abs(before_pos[auditor.geom_ids] - data.geom_xpos[auditor.geom_ids])) < 1e-7
    assert np.max(np.abs(before_mat[auditor.geom_ids] - data.geom_xmat[auditor.geom_ids])) < 1e-7
    rows = []
    for frame, pose in enumerate(poses):
        rate=float(cfg['fps']) if cfg.get('validated_video_pipeline') else 40.
        row = auditor.inspect(pose); row.update(control_frame=frame, source_time_s=frame / rate)
        rows.append(row)
        if frame % 40 == 0:
            print('collision audit', frame, '/', len(poses), 'max depth mm', row['maximum_penetration_m'] * 1000, flush=True)
    maximum = max(row['maximum_penetration_m'] for row in rows)
    failed = [row['control_frame'] for row in rows if row['penetrating_pair_count']]
    result = {'schema_version': 1, 'episode': cfg['episode'], 'states_sha256': hashlib.sha256(states_path.read_bytes()).hexdigest(),
              'geometry_sha256': auditor.geometry_sha256, 'mujoco_version': mujoco.__version__,
              'distance_backend': 'MuJoCo native CCD; geometry queries only, no physics integration',
              'all_recorded_frames_checked': True, 'recorded_frame_count': len(poses),
              'all_required_pairs_checked': True, 'categories_checked': sorted(auditor.categories),
              'collision_solids_per_group': auditor.groups, 'pairs_per_frame': len(auditor.pairs),
              'numerical_epsilon_m': NUMERICAL_EPSILON_M, 'maximum_penetration_m': maximum,
              'maximum_penetration_by_category_m': {kind:max(row['maximum_by_category_m'][kind] for row in rows) for kind in auditor.categories},
              'failed_control_frames': failed, 'failed_frame_count': len(failed),
              'no_penetration_verified': not failed, 'diagnostic_only': True,
              'policy_states_modified': False, 'physics_steps_executed': 0,
              'scope': 'Every recorded source-clock pose: hands versus objects, other hand and environment; objects versus other objects and environment. Same-hand joint assemblies and pieces of the same rigid object are excluded.',
              'limitations': ['Collision solids approximate rendered surfaces.', 'Unrecorded physics substeps are not certified by this replay.'],
              'frames': rows}
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(result, indent=2))
    print(json.dumps({key: value for key, value in result.items() if key != 'frames'}), flush=True)
    raise SystemExit(0)


runner.run_training = inspect
sys.argv = [sys.argv[0], 'experiment=sharpa_flash_sac',
            f"task.parquet='/episode/human_motion_data/{reference}/processed/sequence_id={cfg['sequence']}/robot_name=sharpa_wave'",
            f'task.motion_speed={motion_speed(cfg)}', 'reset.reset_finger_openness=0.0', 'scene.world_count=1',
            'logging.mode=disabled', 'output_dir=/episode/diagnostic-checkpoints',
            'hydra.run.dir=/episode/hydra/recorded-collision-audit']
if cfg.get('predecomposed_object_collisions'):
    sys.argv.append('scene.decompose_objects=false')
if cfg.get('validated_video_pipeline'):sys.argv.append('task.source_frame_playback=true')
runpy.run_path('/workspace/scripts/train_flash_sac.py', run_name='__main__')
