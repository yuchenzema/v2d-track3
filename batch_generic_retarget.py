"""Install per-video multi-object, multi-grasp retargeting and native scripts."""
from pathlib import Path
import json,sys
from legacy_source_frame_playback import prepare_script
from batch_retarget_templates import install as pot_install
BASE=Path('/home/yu_eli/v2d-track3');OLD=BASE/'runs/episode_000005_newton_20261015/scripts'
def install(r):
 c=json.loads((r/'episode-config.json').read_text());T=c['frames'];names=c['objects'];frozen_training=(r/'coherent-input-checksums.json').exists()
 if c['episode'] in [3,6] and not c.get('own_geometry_required'):pot_install(r)
 else:
  s=(OLD/'retarget_hands.py').read_text().replace('episode_000005',c['sequence']);s=s.replace("names=['white_pot','wooden_spoon'];",f'names={names!r};')
  start=s.index('handle_indices=');end=s.index("cells={'fps'",start)
  s=s[:start]+'''config=json.loads((ROOT/'episode-config.json').read_text());grasps=config['grasps']
handle_indices=[np.arange(len(m.vertices)) for m in meshes]
trees=[cKDTree(m.vertices) for m in meshes]
targets=hands.copy();contact_constraints={}
for gi,g in enumerate(grasps):
 side=['left','right'].index(g['side']);b=g['object'];start,end=g['start'],g['end'];lo=max(0,start-5);hi=min(T-1,end+5)
 verts=meshes[b].vertices;normals=meshes[b].vertex_normals;indices=np.arange(start,end+1)
 if config.get('quality_training'):
  from source_grasp_phase import template_frames
  examples,_=template_frames(grasps,g);indices=np.asarray(examples)
 local=np.einsum('nij,nkj->nki',objects[indices,b,:3,:3].transpose(0,2,1),hands[indices,side]-objects[indices,b,None,:3,3]);typical=np.median(local,axis=0)
 opening=typical[4]-np.mean(typical[[8,12,16,20]],axis=0);opening/=np.linalg.norm(opening)+1e-12
 for j in [4,8,12,16,20]:
  align=normals@opening;allowed=np.flatnonzero(align>(.1 if j==4 else -np.inf)) if j==4 else np.flatnonzero(align<-.1)
  if not len(allowed):allowed=np.arange(len(verts))
  idx=allowed[np.argmin(np.linalg.norm(verts[allowed]-typical[j],axis=1))];anchor=verts[idx]+normals[idx]*.0015;contact_constraints[f'grasp_{gi}_{g["side"]}_joint_{j}']=anchor.tolist()
  for i in range(lo,hi+1):
   blend=min(1.,(i-lo)/max(1,start-lo),(hi-i)/max(1,hi-end));blend=max(0.,blend)
   target=objects[i,b,:3,:3]@anchor+objects[i,b,:3,3];targets[i,side,j]=blend*target+(1-blend)*targets[i,side,j]
''' +s[end:]
  s=s.replace('.reshape(T,2,4)','.reshape(T,len(names),4)')
  s=s.replace('held=(22<=i<=249) if s==0 else (37<=i<=268)',"held=any(g['side']==side and g['start']<=i<=g['end'] for g in grasps)")
  s=s.replace('active=(18<=i<=255) if s==0 else (33<=i<=274)',"active=any(g['side']==side and max(0,g['start']-4)<=i<=min(T-1,g['end']+4) for g in grasps)")
  s=s.replace('for b in range(2):','for b in range(len(names)):')
  s=s.replace('            if s==1 and j>2:continue\n            b=s;d=distances[b][j]',"            choices=[g['object'] for g in grasps if g['side']==side and max(0,g['start']-4)<=i<=min(T-1,g['end']+4)]\n            b=min(choices,key=lambda b:distances[b][j]);d=distances[b][j]")
  s=s.replace('post-IK fingertip proximity to the video-grasped handle only; 25 mm maximum gap; right thumb/index/middle grasp; episode 5 observed intervals','Post-IK fingertips within 25 mm of the particular object grasped by this hand in the reviewed source interval')
  s=s.replace('video-observed sustained grasps; left thumb above pot handle and fingers below; right thumb above and fingers below the reconstructed spoon shaft; local contact anchors inferred from estimated human fingertips','Video-observed separate grasps and handovers; local contacts inferred from the human fingertips in this interval and opposing surface normals')
  (r/'scripts/retarget_hands.py').write_text(s)
  s=(BASE/'batch_coherent.py').read_text();(r/'scripts/coherent_grasp_variant.py').write_text(s)
  (r/'scripts/source_grasp_phase.py').write_text((BASE/'source_grasp_phase.py').read_text())
  # Shared native code contains no assumption about a two-object scene.
  for name in ['view_policy_single_attempt.py','record_policy_states.py','recorded_trajectory_renderer.py','render_recorded_oblique.py','render_source_camera.py','assess_single_attempt.py','wrist_velocity_controller.py','record_velocity_controller.py','evaluate_velocity_controller.py']:
   s=(OLD/name).read_text().replace('episode_000005',c['sequence']).replace("name=['white_pot','wooden_spoon'][b]","name=binding.name")
   if name=='view_policy_single_attempt.py':
    s=prepare_script(s,c)
    # CoACD disables collision on the original collision source but keeps that
    # hidden source in the model. Only the visible scan has the public UV mesh.
    s=s.replace("if flags[shape]&int(newton.ShapeFlags.COLLIDE_SHAPES) or source is None", "if not flags[shape]&int(newton.ShapeFlags.VISIBLE) or flags[shape]&int(newton.ShapeFlags.COLLIDE_SHAPES) or source is None")
   (r/'scripts'/name).write_text(s)
  # The transferred actor is only valid when the observation schema matches.
  s=(BASE/'runs/episode_000003_newton_batch_20261015/scripts/train_coherent_actor_transfer.py').read_text().replace('episode_000003',c['sequence']).replace('all 313 source frames',f'all {T} source frames')
  s=s.replace('scene.world_count=2048','scene.world_count=1024').replace('training.replay.capacity=524288','training.replay.capacity=262144').replace('training.replay.minimum_size=51200','training.replay.minimum_size=25600').replace("'world_count':2048","'world_count':1024")
  s=s.replace('Episode 5 policy adaptation completed.','Policy adaptation completed.').replace('public pot asset and video-reconstructed spoon','released per-object assets and an episode-specific video spoon mesh when needed')
  if len(names)==1:
   # A fresh single-object learner keeps schema checking enabled throughout.
   a=s.index('original_restore=');b=s.index('sys.argv=',a);s=s[:a]+s[b:]
   s=s.replace('START=250003456;BUDGET=100003840','START=0;BUDGET=100003840')
   s=s.replace("'training.resume=true','training.checkpoint_path=/prior/state_250003456.safetensors',",'')
   s=s.replace("'preferred_prior_policy':'episode_000001_newton_v2_20261014/flash_chord/state_250003456.safetensors'","'preferred_prior_policy':None")
  if not frozen_training:(r/'scripts/train_coherent_actor_transfer.py').write_text(s)
 # Reusable frame-exact metrics and assets for all cardinalities.
 for name in ['source_geometry.py','assess_source_fidelity.py']:
  s=(OLD/name).read_text().replace('episode_000005',c['sequence']).replace("T=288;NAMES=['white_pot','wooden_spoon']",f'T={T};NAMES={names!r}')
  if c.get('frame_cache_format')=='jpg':s=s.replace('frames/{i:06d}.png','frames/{i:06d}.jpg')
  if name=='assess_source_fidelity.py':
   a=s.index('canonical=np.array(');b=s.index('writer=',a);s=s[:a]+"head_errors=[]\n"+s[b:]
   a=s.index(' P=C@Ms[1]');b=s.index(' if state:',a);s=s[:a]+s[b:]
   s=s.replace('for b in range(2)]','for b in range(len(NAMES))]')
   s=s.replace('  if b==0:valid&=~masks[1]',"  for other in range(len(NAMES)):\n   if other!=b:valid&=~masks[other]")
   s=s.replace(",'mean_spoon_head_projection_error_px':float(np.mean(head_errors))",'')
   s=s.replace(",'mean_spoon_head_error_px':float(np.mean(head_errors[phase['start_frame']:phase['end_frame']+1]))",'')
  if name=='assess_source_fidelity.py':
   s=s.replace('for i in range(T):',"source_reader=cv2.VideoCapture(str(REC/'calibrated.mp4'))\nfor i in range(T):",1)
   for extension in ['png','jpg']:s=s.replace("image=cv2.imread(str(REC/f'frames/{i:06d}."+extension+"'))","ok,image=source_reader.read();assert ok")
  (r/'scripts'/name).write_text(s)
 boundaries=sorted(set([0,T]+[x for g in c['grasps'] for x in [g['start'],g['end']+1]]+list(range(40,T,40))))
 phases=[{'name':f'frames_{lo}_{hi-1}','start_frame':lo,'end_frame':hi-1} for lo,hi in zip(boundaries[:-1],boundaries[1:])]
 (r/'research/source-video-review.json').write_text(json.dumps({'all_frames_visually_reviewed':c['all_frames_visually_reviewed'],'frame_count':T,'grasps':c['grasps'],'notes':c.get('source_phase_notes',c.get('source_review_notes','')),'phases':phases},indent=2))
 if not frozen_training and int(c.get('training_world_count',2048 if c['episode'] in [3,6,8,10,19,20,25,32,33] else 1024))==2048:
  p=r/'scripts/train_coherent_actor_transfer.py';s=p.read_text().replace('scene.world_count=1024','scene.world_count=2048').replace('training.replay.capacity=262144','training.replay.capacity=524288').replace('training.replay.minimum_size=25600','training.replay.minimum_size=51200').replace("'world_count':1024","'world_count':2048");p.write_text(s)
 if not frozen_training:
  p=r/'scripts/train_coherent_actor_transfer.py';s=p.read_text().replace('track3_coherent_variant',c.get('training_reference','track3_coherent_variant'));p.write_text(s)
 (r/'scripts/evaluate_reference_controller.py').write_text((BASE/'evaluate_reference_controller.py').read_text())
 print(c['sequence'],'native scripts installed',flush=True)
if __name__=='__main__':
 for ep in map(int,sys.argv[1:]):install(BASE/'runs'/f'episode_{ep:06d}_newton_batch_20261015')
