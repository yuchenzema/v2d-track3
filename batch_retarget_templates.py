"""Retarget the pot/spoon clips using each clip's observed grasp intervals."""
from pathlib import Path
import json
BASE=Path('/home/yu_eli/v2d-track3');OLD=BASE/'runs/episode_000005_newton_20261015/scripts'
def install(root):
 c=json.loads((root/'episode-config.json').read_text());g=c['grasps'];assert c['objects']==['white_pot','wooden_spoon'];T=c['frames'];a,b=g;assert a['side']=='left' and b['side']=='right'
 spans=[(q['start'],q['end'],max(0,q['start']-8),min(T-1,q['end']+10),q['start']+10,q['end']-10) for q in g]
 s=(OLD/'retarget_hands.py').read_text().replace('episode_000005',c['sequence']);s=s.replace('[(22,249,14,259,40,240),(37,268,29,277,50,215)]',repr(spans))
 s=s.replace('(22<=i<=249) if s==0 else (37<=i<=268)',f"({a['start']}<=i<={a['end']}) if s==0 else ({b['start']}<=i<={b['end']})")
 s=s.replace('(18<=i<=255) if s==0 else (33<=i<=274)',f"({max(0,a['start']-4)}<=i<={a['end']+6}) if s==0 else ({max(0,b['start']-4)}<=i<={b['end']+6})")
 s=s.replace('episode 5 observed intervals','episode-specific video-observed intervals');(root/'scripts/retarget_hands.py').write_text(s)
 s=(OLD/'coherent_grasp_variant.py').read_text().replace('episode_000005',c['sequence']);s=s.replace('[(22,249,14,259),(37,268,29,277)]',repr([x[:4] for x in spans]));(root/'scripts/coherent_grasp_variant.py').write_text(s)
 for name in ['view_policy_single_attempt.py','record_policy_states.py','recorded_trajectory_renderer.py','render_recorded_oblique.py','render_source_camera.py','assess_single_attempt.py','wrist_velocity_controller.py','record_velocity_controller.py','evaluate_velocity_controller.py']:
  s=(OLD/name).read_text().replace('episode_000005',c['sequence']);s=s.replace("name=['white_pot','wooden_spoon'][b]","name=binding.name")
  (root/'scripts'/name).write_text(s)
 s=(OLD/'train_coherent_actor_transfer.py').read_text().replace('episode_000005',c['sequence']).replace('all 288 source frames',f'all {T} source frames').replace('scene.world_count=4096','scene.world_count=1024').replace('training.replay.capacity=1048576','training.replay.capacity=262144').replace('training.replay.minimum_size=102400','training.replay.minimum_size=25600')
 s=s.replace("'world_count':4096","'world_count':1024")
 (root/'scripts/train_coherent_actor_transfer.py').write_text(s)
 print(c['sequence'],'installed per-episode retarget and native policy scripts')
if __name__=='__main__':
 import sys
 for value in sys.argv[1:]:install(BASE/'runs'/f'episode_{int(value):06d}_newton_batch_20261015')
