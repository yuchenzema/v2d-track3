"""Admit the source-contact-resolved handle pose and rebuild its robot references."""
from pathlib import Path
import fcntl,hashlib,json,shutil,subprocess,time
import numpy as np,trimesh
from batch_worker import BASE,PY
from batch_reconstruction_queue import process
from batch_collision_reference import prepare
from batch_refine_contacts import refine
r=BASE/'runs/episode_000028_newton_batch_20261015';rec=r/'reconstruction';cfg=json.loads((r/'episode-config.json').read_text())
with (r/'research/collision-reference.lock').open('a') as lock:
 fcntl.flock(lock,fcntl.LOCK_EX);assert not (r/'coherent-input-checksums.json').exists();candidate=rec/'scene_motion_handle_candidate.npz';new=np.load(candidate);handle=json.loads((r/'research/saucepan-handle-candidate.json').read_text());contact=json.loads((r/'research/saucepan-handle-source-contact-check.json').read_text());before=json.loads((r/'evaluation/physical-reference-source-fidelity.json').read_text())['aggregate']['white_pot']['mean_iou'];after=json.loads((r/'evaluation/handle-candidate-source-fidelity.json').read_text())['aggregate']['objects']['white_pot']['iou'];assert after>=before-.055 and handle['maximum_orientation_step_rad']<=.31;assert contact['aggregate']['right']['after_px']<contact['aggregate']['right']['before_px']*.2
 backup=r/f'research/accepted-saucepan-handle-{time.time_ns()}';backup.mkdir()
 for name in ['scene_motion_physical.npz','pose-continuity.json','rebuilt_scene.glb']:shutil.copy2(rec/name,backup/name)
 shutil.copy2(r/'episode-config.json',backup/'episode-config.json')
 for name in ['track3_video','track3_coherent_variant','track3_collision_variant']:
  path=r/'human_motion_data'/name
  if path.exists():path.rename(backup/name)
 names=['collision-reference-selection.json','contact-preparation-failure.json','collision-aware-grasps.json','collision-aware-applied-grasps.json','collision-transition-poses.json','collision-transition-applied.json','native-object-contact-projection.json','native-object-6d-refinement.json','collision-floor-projection.json','release-contact-targets.json']
 names += [p.name for p in (r/'research').glob('continuous-release-selection*.json')]
 for name in names:
  path=r/'research'/name
  if path.exists():path.rename(backup/name)
 for path in list((r/'evaluation').glob('reference-source-fidelity*'))+list((r/'evaluation').glob('*collision-audit.json'))+list((r/'evaluation').glob('physical-reference-*')):path.rename(backup/path.name)
 shutil.copy2(candidate,rec/'scene_motion_physical.npz');continuity=json.loads((backup/'pose-continuity.json').read_text());continuity['source_contact_resolved_handle_orientation']='research/saucepan-handle-candidate.json';continuity['objects']['white_pot']['maximum_rotation_step_rad']=handle['maximum_orientation_step_rad'];(rec/'pose-continuity.json').write_text(json.dumps(continuity,indent=2));shutil.copy2(r/'evaluation/handle-candidate-source-fidelity.json',r/'evaluation/reference-source-fidelity.json')
 for path in (r/'evaluation').glob('handle-candidate-source-fidelity-review-*.jpg'):shutil.copy2(path,r/'evaluation'/path.name.replace('handle-candidate-source-fidelity','reference-source-fidelity'))
 scene=trimesh.load(rec/'rebuilt_scene.glb');scene.graph.update(frame_to='white_pot',matrix=new['object_to_world'][0,1]);scene.export(rec/'rebuilt_scene.glb')
 # One continuous right grasp has three postures as the wrist rolls the pot.
 # Adjacent intervals have no release frame; the template generator blends them.
 cfg['grasps']=[g for g in cfg['grasps'] if g['side']!='right']+[{'side':'right','object':1,'start':a,'end':b,'continuous_grasp_posture_phase':True} for a,b in [(49,90),(91,130),(131,205)]];cfg['reference_review_pending']=True;cfg['reconstruction_visual_reviewed']=False;cfg['reconstruction_refinement_pending']='Rebuilding source-contact-resolved saucepan and continuous wrist-roll postures';cfg['source_review_notes']+=' The right grasp is continuous from frame 49 through 205; adjacent intervals describe changing wrist posture during inversion and do not indicate releases.';(r/'episode-config.json').write_text(json.dumps(cfg,indent=2));decision={'method':__doc__,'source_body_iou_before':before,'source_body_iou_candidate':after,'source_handle_contact_projection':contact['aggregate'],'maximum_orientation_step_rad':handle['maximum_orientation_step_rad'],'candidate_sha256':hashlib.sha256(candidate.read_bytes()).hexdigest(),'earlier_reconstruction_and_references_preserved':str(backup.relative_to(r)),'candidate_admitted_for_robot_reference_validation':True};(r/'research/saucepan-handle-selection.json').write_text(json.dumps(decision,indent=2));print(decision,flush=True)
process(28);prepare(28);refine(28)
subprocess.run([str(Path('/usr/bin/python3')),'-u',str(BASE/'batch_refine_object_poses.py'),'28'],check=True)
subprocess.run([str(Path('/usr/bin/python3')),'-u',str(BASE/'batch_release_refinement.py'),'28'],check=True)
subprocess.run([str(Path('/usr/bin/python3')),'-u',str(BASE/'batch_finalize_release_references.py'),'28'],check=True)
subprocess.run([PY,str(BASE/'assess_physical_reference.py'),'--run',str(r),'--overlay'],check=True)
print('Episode 28 rebuilt. Manual source/hand review and passive-rest simulation remain required.',flush=True)
