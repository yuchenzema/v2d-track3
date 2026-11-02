"""Run static native collision/FK probes on CPU, preserving the actual MuJoCo model.

These probes call mj_forward directly and never advance the physics simulation.
Every frame and collision hull matched the CUDA-backed setup exactly in the
235-frame episode 19 equivalence check.
"""
import runpy,sys
from flash_chord.envs.rl import RLEnv

RLEnv.capture_step=lambda self:None
target=sys.argv[1];sys.argv=[target,*sys.argv[2:]]
original=runpy.run_path
def run_path(path,*args,**kwargs):
 if str(path)=='/workspace/scripts/train_flash_sac.py':
  assert not any(x.startswith('training.device=') for x in sys.argv)
  sys.argv.append('training.device=cpu')
 return original(path,*args,**kwargs)
runpy.run_path=run_path
original(target,run_name='__main__')
