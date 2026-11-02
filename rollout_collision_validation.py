"""Bind independent recorded-pose collision checks to the exact rollout artifact."""
import hashlib
import json
import os
from pathlib import Path
import shutil
from contextlib import nullcontext
from improvement_common import BASE, BATCH, save, utc, digest
from episode_execution_guard import owner_alive, episode_slot

COLLISION_EPSILON_M = 1e-6


def collision_verified(report, states_sha256, version=None, frame_count=None):
    try:
        return bool(report.get('all_recorded_frames_checked') is True
            and report.get('all_required_pairs_checked') is True
            and report.get('no_penetration_verified') is True
            and report.get('failed_frame_count') == 0
            and report.get('states_sha256') == states_sha256 and states_sha256
            and (version is None or report.get('version') == version)
            and isinstance(report.get('recorded_frame_count'), int) and report['recorded_frame_count'] > 0
            and (frame_count is None or report['recorded_frame_count'] == frame_count)
            and report.get('geometry_sha256')
            and 0 <= report['maximum_penetration_m'] <= report['numerical_epsilon_m'] <= COLLISION_EPSILON_M)
    except (KeyError, TypeError, ValueError):
        return False


def audit_rollout(root, states, version, reference=None):
    """Keep all GPU work unchanged; the replay runs on CPU in the same episode."""
    from batch_native import _docker
    root, states = Path(root).resolve(), Path(states).resolve()
    assert states.is_relative_to(root), 'Collision audit states must belong to this run'
    cfg = json.loads((root / 'episode-config.json').read_text())
    state_hash = digest(states)
    helpers = ['rigid_pose_collision_audit.py', 'audit_recorded_collisions_native.py', 'run_geometry_cpu.py',
               'episode_clock.py','legacy_source_frame_playback.py']
    reference = reference or cfg.get('training_reference', 'track3_coherent_variant')
    inputs = [root / 'episode-config.json', *[BASE / name for name in helpers],
              *sorted((root / f'human_motion_data/{reference}/processed').rglob('motion.parquet')),
              *sorted((root / 'reconstruction/mesh_priors').glob('*.ply')),
              *sorted((root / 'reconstruction/mesh_priors').glob('*.obj')),
              *sorted((root / 'human_motion_data' / reference).rglob('*.usda'))]
    signature = hashlib.sha256(json.dumps([(str(p), digest(p)) for p in inputs]).encode()).hexdigest()
    path = root / f'evaluation/collision-{state_hash[:20]}-{signature[:12]}.json'
    if path.exists():
        cached = json.loads(path.read_text())
        if cached.get('states_sha256') == state_hash and cached.get('input_signature') == signature:
            cached['version'] = version
            save(path,cached)
            compact={key:value for key,value in cached.items() if key!='frames'}
            save(root/'research/rollout-collision-validation.json',compact)
            return compact
    paused = json.loads((BATCH / 'user-pause.json').read_text()) if (BATCH / 'user-pause.json').exists() else {}
    if paused.get('status') == 'paused_by_user':
        return {'no_penetration_verified': False, 'status': 'paused_by_user'}
    owner = json.loads((BATCH / 'episode-slot-owner.json').read_text()) if (BATCH / 'episode-slot-owner.json').exists() else {}
    busy = owner_alive(owner)
    if busy and owner.get('episode') != cfg['episode']:
        return {'no_penetration_verified': False, 'status': 'waiting_for_current_episode', 'episode': cfg['episode']}
    context = nullcontext() if busy else episode_slot(root, cfg['episode'])
    with context:
        for name in helpers:
            shutil.copy2(BASE / name, root / 'scripts' / name)
        native = cfg.get('native_container')
        assert native, 'Native scene container is required for collision validation'
        info = json.loads(_docker(['inspect', native]).stdout)[0]
        assert any(mount.get('Source') == str(root) and mount.get('Destination') == '/episode' for mount in info.get('Mounts', [])), 'Native container belongs to another episode run'
        if not info['State']['Running']:
            _docker(['start', native])
        inside_states = '/episode/' + str(states.relative_to(root))
        inside_output = '/episode/' + str(path.relative_to(root))
        _docker(['exec', native, 'env', 'CUDA_VISIBLE_DEVICES=', 'JAX_PLATFORMS=cpu',
                 'OPENBLAS_NUM_THREADS=2', 'OMP_NUM_THREADS=2',
                 'V2D_COLLISION_STATES=' + inside_states, 'V2D_COLLISION_OUTPUT=' + inside_output,
                 'V2D_COLLISION_REFERENCE=' + reference, 'python', '-u',
                 '/episode/scripts/run_geometry_cpu.py', '/episode/scripts/audit_recorded_collisions_native.py'],
                root / 'logs/recorded-collision-audit.log')
    report = json.loads(path.read_text())
    assert digest(states) == state_hash, 'Rollout states changed during collision validation'
    assert report['states_sha256'] == state_hash
    report.update(version=version, updated_utc=utc(), input_signature=signature,
                  recorded_states=str(states),
                  report='/artifacts/' + str(path.relative_to(BASE / 'runs')))
    save(path, report)
    compact = {key: value for key, value in report.items() if key != 'frames'}
    save(root / 'research/rollout-collision-validation.json', compact)
    return compact
