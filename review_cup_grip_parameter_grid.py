"""Search finger motor calibration, then validate uniform controllers independently.

The discovery cohort uses declared different parameters per world. Its worlds
are never selected for publication. Selected parameters receive fresh uniform
32-world evaluations and the usual predetermined single-attempt recording.
"""
import hashlib,importlib.util,json,shutil
from pathlib import Path
import cv2,numpy as np,trimesh
from improvement_common import BASE,save,baseline,better,verify_preserved
from batch_native import docker
from batch_review import review
R=BASE/'runs/episode_000010_newton_grasp_curriculum_20261020'
alternate=R/'research/precision-schedule-checkpoint-360000000/policy_360000000.safetensors'
actors={'alternate_checkpoint':'/episode/'+str(alternate.relative_to(R)),'alternate_checkpoint_sha256':hashlib.sha256(alternate.read_bytes()).hexdigest(),'hand_weights':{'left':.5,'right':0.}}
profiles=[{'left':round(x,3),'right':right} for right in [0.,.025] for x in np.linspace(0,.3,13)]
profiles += [{'left':x,'right':.05} for x in [0.,.05,.1,.15,.2,.25]];assert len(profiles)==32
discovery={**actors,'finger_bias_profiles':profiles};state=R/'research/cup-grip-grid-states.npz'
for name in ['bimanual_policy_controller.py','grip_bias_controller.py','record_native_cohort.py','evaluate_reference_controller.py']:shutil.copy2(BASE/name,R/'scripts'/name)
if not state.exists():
 docker(['exec','v2d-grasp-episode10-newton','env','V2D_RECORD_COHORT_SIZE=32','V2D_WRIST_VELOCITY_GAIN=0',
  'V2D_REFERENCE_OVERRIDE=/episode/human_motion_data/track3_collision_variant/processed/sequence_id=episode_000010/robot_name=sharpa_wave',
  'V2D_SIM_SUBSTEPS=10','V2D_SIM_NCONMAX=2048','V2D_SIM_NJMAX=4096','V2D_ACTOR_COMBINATION='+json.dumps(discovery,sort_keys=True),
  'V2D_STATE_OUTPUT=/episode/research/cup-grip-grid-states.npz','python','-u','/episode/scripts/record_native_cohort.py',
  'evaluation.checkpoint=/episode/research/initial-policy/policy_350007296.safetensors','evaluation.world_count=32','logging.mode=disabled',
  'evaluation.metrics_output=/episode/research/cup-grip-grid-native-metrics.json','hydra.run.dir=/episode/hydra/cup-grip-grid'],R/'logs/cup-grip-grid.log')
data=np.load(state.with_name(state.stem+'-cohort-objects.npz'));poses=data['achieved'];assert poses.shape[1:3]==(32,2)
spec=importlib.util.spec_from_file_location('grid_source_geometry',R/'scripts/source_geometry.py');g=importlib.util.module_from_spec(spec);spec.loader.exec_module(g)
cfg=json.loads((R/'episode-config.json').read_text());motion=np.load(R/'reconstruction/scene_motion_physical.npz');measure=np.load(R/'reconstruction/hand_measurements.npz')
meshes=[trimesh.load(R/f'reconstruction/mesh_priors/{n}.ply',process=False) for n in cfg['objects']];T=cfg['frames'];scores=np.zeros((32,T,2));finite=np.isfinite(poses).all(axis=(0,2,3))
for i in range(T):
 C=np.linalg.inv(motion['camera_to_world'][i]);exclude=g.human_occlusion(measure,i)
 masks=[cv2.resize(cv2.imread(str(R/f'reconstruction/masks/{b+1}/{i:06d}.png'),0),(512,384),interpolation=cv2.INTER_NEAREST)>0 for b in range(2)]
 frame=round(i*(len(poses)-1)/(T-1))
 for world in np.flatnonzero(finite):
  for b in range(2):
   mask=g.silhouette(meshes[b].vertices,C@g.matrix(poses[frame,world,b]));scores[world,i,b]=g.overlap(mask,masks[b],(~exclude)&(~masks[1-b]))['iou']
rows=[]
for world,profile in enumerate(profiles):
 held=[];transition=[]
 for b in range(2):
  grasps=[v for v in cfg['grasps'] if v['object']==b];frames=sorted({i for v in grasps for i in range(v['start'],v['end']+1)})
  boundaries=sorted({i for v in grasps for boundary in [v['start'],v['end']] for i in range(max(0,boundary-4),min(T,boundary+5))})
  held.append(float(scores[world,frames,b].mean()));transition.append(float(scores[world,boundaries,b].mean()))
 score=.5*np.mean(held)+.25*min(held)+.25*np.mean(transition)
 rows.append({'diagnostic_world_index':world,'finger_bias_rad':profile,'finite':bool(finite[world]),'source_mean_iou':float(scores[world].mean()),
  'held_iou_by_object':dict(zip(cfg['objects'],held)),'ranking_score':float(score)})
ranked=sorted([r for r in rows if r['finite']],key=lambda r:r['ranking_score'],reverse=True);selected=ranked[:3]
save(R/'research/cup-grip-grid-results.json',{'diagnostic_only':True,'discovery_profiles':rows,'uniform_revalidation_candidates':selected,
 'discovery_worlds_are_not_eligible_for_publication':True,'all_original_source_frames_checked':T})
print('Grip grid top candidates',selected,flush=True)
for candidate in selected:
 profile=candidate['finger_bias_rad'];uniform={**actors,'finger_bias_rad':profile};label=f"left{profile['left']}-right{profile['right']}"
 row=review(10,350007296,root_override=R,container_override='v2d-grasp-episode10-newton',
  checkpoint_override='/episode/research/initial-policy/policy_350007296.safetensors',reference_override='track3_collision_variant',actor_combination=uniform)
 row['passes_baseline_improvement_gate']=better(row,baseline(10));save(R/f'research/grip-grid-uniform-{label}-review.json',row)
 print('Uniform revalidation',label,{k:row[k] for k in ['source_mean_object_iou','completion_rate_32_worlds','passes_baseline_improvement_gate']},flush=True);verify_preserved(10)
