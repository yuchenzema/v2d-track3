"""Finish source-image object-pose refinement once each contact reference is ready."""
from pathlib import Path
import subprocess,sys,time
base=Path('/home/yu_eli/v2d-track3')
for ep in map(int,sys.argv[1:]):
 r=base/'runs'/f'episode_{ep:06d}_newton_batch_20261015'
 while not (r/'research/collision-reference-selection.json').exists():time.sleep(15)
 assert not (r/'coherent-input-checksums.json').exists()
 subprocess.run([sys.executable,'-u',str(base/'batch_refine_object_poses.py'),str(ep)],cwd=base,check=True)
