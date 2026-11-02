"""Promote a reviewed, quantitatively improved reconstruction before any training starts."""
from pathlib import Path
import fcntl,hashlib,json,shutil,sys,time
import numpy as np,trimesh
from batch_reconstruction_queue import process
BASE=Path('/home/yu_eli/v2d-track3')
for ep in map(int,sys.argv[1:]):
 r=BASE/'runs'/f'episode_{ep:06d}_newton_batch_20261015';rec=r/'reconstruction'
 with (r/'research/collision-reference.lock').open('a') as lock:
  fcntl.flock(lock,fcntl.LOCK_EX);assert not (r/'coherent-input-checksums.json').exists()
  marker=r/'research/image-geometry-selection.json'
  if marker.exists():continue
  candidate=rec/'scene_motion_image_geometry_candidate.npz';data=np.load(candidate);assert all(np.isfinite(data[k]).all() for k in data.files)
  old=json.loads((r/'evaluation/reference-source-fidelity.json').read_text())['aggregate']['objects']['wooden_spoon'];new=json.loads((r/'evaluation/image-geometry-regularized.json').read_text())['aggregate']['objects']['wooden_spoon']
  assert new['iou']>old['iou']+.025 and new['centroid_error_px']<old['centroid_error_px'],(old,new)
  backup=r/f'research/accepted-image-geometry-{time.time_ns()}';backup.mkdir()
  for name in ['scene_motion_physical.npz','pose-continuity.json']:
   p=rec/name
   if p.exists():shutil.copy2(p,backup/name)
  for name in ['track3_video','track3_coherent_variant','track3_collision_variant']:
   p=r/'human_motion_data'/name
   if p.exists():p.rename(backup/name)
  for p in list((r/'evaluation').glob('reference-source-fidelity*')):p.rename(backup/p.name)
  for name in ['collision-reference-selection.json','contact-preparation-failure.json','native-object-contact-projection.json','collision-aware-grasps.json','collision-transition-poses.json']:
   p=r/'research'/name
   if p.exists():p.rename(backup/name)
  shutil.copy2(candidate,rec/'scene_motion_physical.npz')
  continuity=json.loads((backup/'pose-continuity.json').read_text());updated=json.loads((rec/'image-geometry-pose-continuity.json').read_text());continuity['objects']['wooden_spoon']=updated['objects']['wooden_spoon'];continuity['accepted_image_geometry_candidate']=True;(rec/'pose-continuity.json').write_text(json.dumps(continuity,indent=2))
  shutil.copy2(r/'evaluation/image-geometry-regularized.json',r/'evaluation/reference-source-fidelity.json')
  for p in (r/'evaluation').glob('image-geometry-regularized-review-*.jpg'):shutil.copy2(p,r/'evaluation'/p.name.replace('image-geometry-regularized','reference-source-fidelity'))
  scene=trimesh.load(rec/'rebuilt_scene.glb');cfg=json.loads((r/'episode-config.json').read_text())
  for b,name in enumerate(cfg['objects']):scene.graph.update(frame_to=name,matrix=data['object_to_world'][0,b])
  scene.export(rec/'rebuilt_scene.glb')
  decision={'method':__doc__,'before':old,'after':new,'candidate_sha256':hashlib.sha256(candidate.read_bytes()).hexdigest(),'backup':str(backup.relative_to(r)),'all_source_frames_scored':cfg['frames'],'original_frame_grid_unchanged':True,'occluded_geometry_remains_estimated':True};marker.write_text(json.dumps(decision,indent=2));cfg['reconstruction_visual_reviewed']=True;cfg['reconstruction_review_notes']='Accepted full-frame image-mask/independent-pinch reconstruction after source overlay review; orientation/depth continuity regularized. Earlier raw FoundationPose trajectory and retargeted references preserved.';(r/'episode-config.json').write_text(json.dumps(cfg,indent=2))
 process(ep)
 print(ep,'improved tool reconstruction accepted',old,new,flush=True)
