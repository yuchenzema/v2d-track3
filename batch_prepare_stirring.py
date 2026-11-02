from pathlib import Path
import subprocess,sys,time
from batch_collision_reference import prepare
base=Path('/home/yu_eli/v2d-track3')
for ep in [32,33]:
 prepare(ep)
 subprocess.run([sys.executable,'-u',str(base/'batch_refine_object_poses.py'),str(ep)],cwd=base,check=True)
