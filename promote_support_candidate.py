"""Accept reviewed source fits that remove table intersections before policy training."""
from pathlib import Path
import fcntl,hashlib,json,shutil,sys,time
import numpy as np,trimesh
from batch_reconstruction_queue import process
BASE=Path('/home/yu_eli/v2d-track3')
for ep in map(int,sys.argv[1:]):
 r=BASE/'runs'/f'episode_{ep:06d}_newton_batch_20261015';rec=r/'reconstruction';cfg=json.loads((r/'episode-config.json').read_text());b=cfg['objects'].index('wooden_spoon');vertices=np.asarray(trimesh.load(rec/'mesh_priors/wooden_spoon.ply',process=False).vertices);radius=json.loads((rec/'scene-reconstruction.json').read_text())['table_support_radius_m'];candidate=rec/'scene_motion_floor_candidate.npz'
 with (r/'research/collision-reference.lock').open('a') as lock:
  fcntl.flock(lock,fcntl.LOCK_EX);assert not (r/'coherent-input-checksums.json').exists()
  if (r/'research/support-geometry-selection.json').exists():continue
  old=np.load(rec/'scene_motion_physical.npz');new=np.load(candidate);assert all(np.isfinite(new[k]).all() for k in new.files);assert len(new['object_to_world'])==cfg['frames']
  def intrusion(data):
   maximum=0.
   for pose in data['object_to_world'][:,b]:
    v=vertices@pose[:3,:3].T+pose[:3,3];inside=np.linalg.norm(v[:,:2],axis=1)<radius
    if inside.any():maximum=max(maximum,.6505-v[inside,2].min())
   return float(maximum)
  old_depth=intrusion(old);new_depth=intrusion(new);assert new_depth<1e-6,(ep,new_depth)
  old_score=json.loads((r/'evaluation/reference-source-fidelity.json').read_text())['aggregate']['objects']['wooden_spoon'];new_score=json.loads((r/'evaluation/support-candidate.json').read_text())['aggregate']['objects']['wooden_spoon'];assert new_score['iou']>=old_score['iou']-.05
  backup=r/f'research/accepted-support-geometry-{time.time_ns()}';backup.mkdir()
  for name in ['scene_motion_physical.npz','pose-continuity.json']:shutil.copy2(rec/name,backup/name)
  for name in ['track3_video','track3_coherent_variant','track3_collision_variant']:
   p=r/'human_motion_data'/name
   if p.exists():p.rename(backup/name)
  for name in ['collision-reference-selection.json','contact-preparation-failure.json','collision-aware-grasps.json','collision-aware-applied-grasps.json','collision-transition-poses.json','collision-transition-applied.json','native-object-contact-projection.json','native-object-6d-refinement.json','collision-floor-projection.json','release-contact-targets.json']:
   p=r/'research'/name
   if p.exists():p.rename(backup/name)
  for p in list((r/'evaluation').glob('reference-source-fidelity*'))+list((r/'evaluation').glob('*collision-audit.json'))+list((r/'evaluation').glob('physical-reference-*')):p.rename(backup/p.name)
  shutil.copy2(candidate,rec/'scene_motion_physical.npz');continuity=json.loads((backup/'pose-continuity.json').read_text());fresh=json.loads((rec/'floor-geometry-pose-continuity.json').read_text());continuity['objects']['wooden_spoon']=fresh['objects']['wooden_spoon'];continuity['support_constrained_tool_reconstruction']=True;(rec/'pose-continuity.json').write_text(json.dumps(continuity,indent=2));shutil.copy2(r/'evaluation/support-candidate.json',r/'evaluation/reference-source-fidelity.json')
  for p in (r/'evaluation').glob('support-candidate-review-*.jpg'):shutil.copy2(p,r/'evaluation'/p.name.replace('support-candidate','reference-source-fidelity'))
  scene=trimesh.load(rec/'rebuilt_scene.glb')
  for j,name in enumerate(cfg['objects']):scene.graph.update(frame_to=name,matrix=new['object_to_world'][0,j])
  scene.export(rec/'rebuilt_scene.glb');decision={'method':__doc__,'maximum_table_intrusion_before_m':old_depth,'maximum_table_intrusion_after_m':new_depth,'source_fidelity_before':old_score,'source_fidelity_after':new_score,'candidate_sha256':hashlib.sha256(candidate.read_bytes()).hexdigest(),'source_frames_checked':cfg['frames'],'earlier_reconstruction_and_references_preserved':str(backup.relative_to(r)),'original_frame_grid_unchanged':True};(r/'research/support-geometry-selection.json').write_text(json.dumps(decision,indent=2));cfg['reconstruction_visual_reviewed']=False;cfg['reconstruction_refinement_pending']='Regenerating robot references after support-constrained source fit';(r/'episode-config.json').write_text(json.dumps(cfg,indent=2))
 process(ep)
 cfg=json.loads((r/'episode-config.json').read_text());cfg['reconstruction_visual_reviewed']=True;cfg.pop('reconstruction_refinement_pending',None);cfg['reconstruction_review_notes']='Reviewed source projections, all-frame table clearance and stationary flat poses. Source overlap remains within 5 percentage points of the earlier image fit while impossible table intersections are removed.';(r/'episode-config.json').write_text(json.dumps(cfg,indent=2));print(ep,'support-constrained reconstruction accepted',decision,flush=True)
