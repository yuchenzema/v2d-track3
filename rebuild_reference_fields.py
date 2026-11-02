"""Rebuild robot FK/contact fields in an isolated candidate with Isaac's kinematics."""
import argparse,json,shlex,subprocess
from pathlib import Path
from improvement_common import BASE

def rebuild(episode,reference,fit=None,label=None):
 r=BASE/f'runs/episode_{episode:06d}_newton_grasp_curriculum_20261020'
 assert not (r/'coherent-input-checksums.json').exists(),'Training inputs are already frozen'
 import shutil
 shutil.copy2(BASE/'apply_collision_transitions.py',r/'scripts/apply_collision_transitions.py')
 image=subprocess.check_output(['sg','docker','-c',shlex.join(['docker','inspect','v2d-batch-episode32-retarget','--format','{{.Image}}'])],text=True).strip()
 label=label or reference+'-rebuilt-fields'
 args=['docker','run','--rm','--name',f'v2d-candidate-{episode}-fk-cpu','--ipc','host','-e','CUDA_VISIBLE_DEVICES=',
  '-e',f'V2D_TRANSITION_REFERENCE={reference}','-e',f'V2D_REBUILD_ONLY={int(fit is None)}',
  '-e',f'V2D_CONTACT_APPLY_OUTPUT=research/{label}.json',
  '-v',f'{r}:/episode','-v',f'{BASE}/video_to_data/robotic_grounding/source/robotic_grounding/robotic_grounding/assets:/v2d/assets:ro',
  '--entrypoint','/isaac-sim/python.sh',image,'-u','/episode/scripts/apply_collision_transitions.py']
 if fit:args[args.index('--entrypoint'):args.index('--entrypoint')]=['-e',f'V2D_CONTACT_FIT_OUTPUT={fit}']
 with (r/f'logs/{label}.log').open('w') as f:subprocess.run(['sg','docker','-c',shlex.join(args)],stdout=f,stderr=subprocess.STDOUT,check=True)
if __name__=='__main__':
 p=argparse.ArgumentParser();p.add_argument('episode',type=int);p.add_argument('reference');p.add_argument('--fit');a=p.parse_args();rebuild(a.episode,a.reference,a.fit)
