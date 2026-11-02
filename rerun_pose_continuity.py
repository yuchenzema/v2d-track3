"""Refit an untrained episode's orientation branches while preserving earlier attempts."""
from pathlib import Path
import json,shutil,sys,time
import numpy as np,trimesh
from pose_continuity import optimize
BASE=Path('/home/yu_eli/v2d-track3')
for ep in map(int,sys.argv[1:]):
 r=BASE/'runs'/f'episode_{ep:06d}_newton_batch_20261015';rec=r/'reconstruction';assert not (r/'coherent-input-checksums.json').exists(),'Training inputs already frozen';backup=r/f'research/pose-continuity-refit-{time.time_ns()}';backup.mkdir()
 if not (rec/'scene_motion_before_pose_continuity.npz').exists():shutil.copy2(rec/'scene_motion_physical.npz',rec/'scene_motion_before_pose_continuity.npz')
 for name in ['scene_motion_physical.npz','pose-continuity.json']:
  p=rec/name
  if p.exists():p.rename(backup/name)
 shutil.copy2(rec/'scene_motion_before_pose_continuity.npz',rec/'scene_motion_physical.npz');optimize(r)
 for name in ['track3_video','track3_coherent_variant']:
  p=r/'human_motion_data'/name
  if p.exists():p.rename(backup/name)
 for p in (r/'evaluation').glob('reference-source-fidelity*'):p.rename(backup/p.name)
 p=rec/'rebuilt_scene.glb'
 if p.exists():
  scene=trimesh.load(p);motion=np.load(rec/'scene_motion_physical.npz');names=json.loads((r/'episode-config.json').read_text())['objects']
  for b,n in enumerate(names):scene.graph.update(frame_to=n,matrix=motion['object_to_world'][0,b])
  scene.export(p)
 c=json.loads((r/'episode-config.json').read_text());c['reconstruction_visual_reviewed']=False;(r/'episode-config.json').write_text(json.dumps(c,indent=2))
