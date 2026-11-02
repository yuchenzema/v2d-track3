"""Freeze real original artifacts even when an old release JSON was never emitted."""
import json
from pathlib import Path
from improvement_common import BATCH, digest


def original_artifacts(source, episode, _seen=None):
    source=Path(source)
    seen=set() if _seen is None else set(_seen)
    if source.resolve() in seen:raise RuntimeError('Cyclic original baseline provenance')
    seen.add(source.resolve())
    manifest=source/'deliverables/manifest.json'
    entries=json.loads(manifest.read_text()) if manifest.is_file() else {}
    if entries and isinstance(entries,dict) and all(isinstance(row,dict) and 'path' in row and 'sha256' in row for row in entries.values()):
        watched={str(source/row['path']):row['sha256'] for row in entries.values()}
    else:
        legacy=BATCH/f'all20_evaluation/episode_{episode:06d}-release.json'
        if legacy.is_file():
            data=json.loads(legacy.read_text())
            watched={row['path']:row['sha256'] for row in data['files'].values()}
            if data.get('checkpoint_sha256'):watched[data['checkpoint']]=data['checkpoint_sha256']
        else:
            # The absence of a release document must not invent an empty
            # preservation contract. Freeze every actual baseline product.
            files=set()
            for folder in ['deliverables','videos','evaluation','reconstruction/mesh_priors']:
                files.update(p for p in (source/folder).rglob('*') if p.is_file())
            for pattern in ['flash_chord*/policy_*.safetensors','flash_chord*/training_state*.safetensors']:
                files.update(source.glob(pattern))
            for name in ['episode-config.json','result-report.json','final-controller-selection.json',
                         'reconstruction/scene_motion.npz','reconstruction/scene_motion_physical.npz']:
                if (source/name).is_file():files.add(source/name)
            watched={str(p):digest(p) for p in sorted(files)}
            if not any(p.suffix=='.safetensors' for p in files):
                config=source/'episode-config.json'
                baseline=json.loads(config.read_text()).get('baseline_run') if config.is_file() else None
                if not baseline:raise RuntimeError('Original baseline lacks artifacts/checkpoints to preserve')
                watched.update(original_artifacts(Path(baseline),episode,seen))
    for path, expected in watched.items():
        if digest(path)!=expected:raise RuntimeError('Original release artifact changed: '+path)
    return watched
