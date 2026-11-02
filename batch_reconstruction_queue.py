"""Sequential GPU reconstruction with per-episode quality reports and resumable state."""
from pathlib import Path
import json,sys,subprocess,time,traceback,shutil,os
import batch_worker as w
from batch_generic_retarget import install
from batch_native import containers,docker
B=w.BASE/'runs/evaluation_batch_20261015'
def status(r,stage,state,**extra):
 p=r/'pipeline-status.json';c=json.loads(p.read_text()) if p.exists() else {};c[stage]={'status':state,'utc':time.strftime('%Y-%m-%dT%H:%M:%SZ',time.gmtime()),**extra};tmp=p.with_suffix('.tmp');tmp.write_text(json.dumps(c,indent=2));tmp.replace(p)
def process(ep):
 r,c=w.context(ep);assert c['all_frames_visually_reviewed'];status(r,'reconstruction','running')
 if not (r/'reconstruction/scene_motion_physical.npz').exists():
  w.metric(r,c)
  if ep in [3,6]:w.pot_spoon_reconstruction(r,c)
  else:w.generic_reconstruction(r,c)
 install(r)
 if not (r/'evaluation/reference-source-fidelity.json').exists():w.run([w.PY,str(r/'scripts/assess_source_fidelity.py'),'--output','reference-source-fidelity','--overlay'],r/'logs/reference-source-fidelity.log')
 if not (r/'reconstruction/rebuilt_scene.glb').exists():w.run([w.PY,str(w.BASE/'batch_visuals.py'),str(r)],r/'logs/prepare-visuals.log')
 if not (r/f"human_motion_data/track3_coherent_variant/processed/sequence_id={c['sequence']}/robot_name=sharpa_wave/motion.parquet").exists():
  _,retarget=containers(r,ep)
  for name in ['retarget_hands','coherent_grasp_variant']:docker(['exec',retarget,'/isaac-sim/python.sh','-u',f'/episode/scripts/{name}.py'],r/f'logs/{name}.log')
 fidelity=json.loads((r/'evaluation/reference-source-fidelity.json').read_text());scores={n:v['iou'] for n,v in fidelity['aggregate']['objects'].items()};weak=[n for n,v in scores.items() if v<.55]
 status(r,'reconstruction','needs_review' if weak else 'completed',source_object_iou=scores,weak_objects=weak)
 status(r,'retarget','completed');print(ep,'reconstructed',scores,'quality review needed',weak,flush=True)
 # Cached dense maps are reproducible from retained original videos and stage
 # commands. Preserve 12 actual metric maps, all poses, all source review sheets,
 # original clips, masks, calibrated MP4, meshes and native references.
 rec=r/'reconstruction';keep=set(round(i*(c['frames']-1)/11) for i in range(12));removed=[]
 for directory in ['depth_moge','depth_stereo','depth','stereo_valid']:
  folder=rec/directory
  if not folder.exists():continue
  for p in folder.glob('*.png'):
   if int(p.stem) not in keep:
    if os.access(p.parent,os.W_OK):p.unlink();removed.append(str(p.relative_to(r)))
 (r/'research/compact-depth-cache.json').write_text(json.dumps({'retained_metric_frames':sorted(keep),'removed_reproducible_dense_intermediates':len(removed),'all_original_videos_retained':True,'all_camera_object_hand_trajectories_retained':True,'regeneration':'Repeat recorded SAM/MoGE/stereo commands; selected frame existence alone does not imply a complete dense cache.'},indent=2))
 return not weak
if __name__=='__main__':
 for ep in map(int,sys.argv[1:]):
  r,c=w.context(ep)
  try:process(ep)
  except Exception:
   status(r,'reconstruction','failed',traceback=traceback.format_exc());print('FAILED',ep,traceback.format_exc(),flush=True)
