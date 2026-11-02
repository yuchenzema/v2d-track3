"""Validate and publish one complete checkpoint video while its learner continues."""
import argparse,json,shutil,fcntl,time,traceback
from pathlib import Path
from improvement_common import BASE,BATCH,baseline,better,digest,save,utc,verify_preserved
from batch_native import docker
from render_improved_controller import render
TARGETS=[(10,'grasp_curriculum_20261020'),(16,'grasp_curriculum_20261020'),
         (19,'grasp_curriculum_20261020'),(20,'backward_grasp_20261020'),
         (30,'grasp_curriculum_20261020'),(32,'backward_grasp_20261020'),(33,'backward_grasp_20261020')]

def publish(ep,step,variant,result_directory=None):
 r=BASE/'runs'/f'episode_{ep:06d}_newton_{variant}';cfg=json.loads((r/'episode-config.json').read_text())
 row=next(x for x in json.loads((r/'checkpoint-reviews.json').read_text()) if x['environment_steps']==step)
 previous=baseline(ep);assert better(row,previous),'Require measured per-object full-episode improvement'
 assert row['single_attempt_tracking']['all_frames_finite']
 dst=r/f'research/checkpoint-preview-{step}';dst.mkdir(exist_ok=True)
 selected=dict(row)
 for kind in ['policy','state']:
  src=r/f'flash_chord_coherent_actor_transfer/{kind}_{step}.safetensors';target=dst/src.name
  if not target.exists():shutil.copy2(src,target)
  if src.exists():assert digest(src)==digest(target)
 selected['checkpoint']='/episode/'+str((dst/f'policy_{step}.safetensors').relative_to(r))
 assert digest(dst/f'policy_{step}.safetensors')==selected['checkpoint_sha256']
 metrics=dst/'metrics-128.json'
 if not metrics.exists():
  docker(['exec',cfg['native_container'],'env','V2D_WRIST_VELOCITY_GAIN=0.',
   f"V2D_SIM_SUBSTEPS={selected['physics_substeps']}",
   *[f'V2D_SIM_{key.upper()}={value}' for key,value in selected.get('solver_capacity',{}).items()],
   f"V2D_REFERENCE_OVERRIDE=/episode/human_motion_data/{selected['inference_reference']}/processed/sequence_id={cfg['sequence']}/robot_name=sharpa_wave",
   'python','-u','/episode/scripts/evaluate_reference_controller.py',f"evaluation.checkpoint={selected['checkpoint']}",
   'evaluation.world_count=128','logging.mode=disabled','evaluation.metrics_output=/episode/'+str(metrics.relative_to(r)),
   f'hydra.run.dir=/episode/hydra/checkpoint-preview-{step}-128'],r/f'logs/checkpoint-preview-{step}-128.log')
 cohort=json.loads(metrics.read_text());assert cohort['step_count']==2*(cfg['frames']-1) and cohort['start_frame']==0
 baseline128=json.loads((Path(cfg['baseline_run'])/'evaluation/final-metrics.json').read_text())['metrics']['chord_sr']
 assert cohort['metrics']['chord_sr']>=baseline128-.0625,'Larger validation cohort regressed'
 rendered=render(ep,selected,dst,root_override=r,container_override=cfg['native_container'])
 verify_preserved(ep)
 quality={'all_internal_frame_checks_passed':selected['object_motion_quality']['every_object_passes_all_local_frames'] and cohort['metrics']['chord_sr']==1.,
  'completion_rate_against_video_estimate':cohort['metrics']['chord_sr'],
  'source_frames_below_internal_0_5_iou':{n:v['source_frames_below_0_5_iou'] for n,v in selected['object_motion_quality']['objects'].items()},
  'full_human_video_match_verified':False,'training_continues':True}
 result={'episode':ep,'status':'completed','release_type':'checkpoint','updated_utc':utc(),
  'evaluation_run':str(r),'experiment':'Completed full-source checkpoint video; policy training remains in progress',
  'selected':selected,'baseline':previous,'measured_improvement':True,'rendered':rendered,
  'native_128_world_evaluation':cohort,'quality_assessment':quality,
  'training_completed':False,'full_human_motion_match_verified':False}
 output=(Path(result_directory) if result_directory else BATCH/'improvements_20261020_grasp')/f'episode_{ep:06d}-checkpoint-{step}-result.json'
 save(output,result);print('Checkpoint preview verified and published:',output,flush=True)

def watch():
 from video_catalog import catalog
 from object_motion_quality import attach
 out=BATCH/'improvements_20261020_grasp'
 with (out/'checkpoint-preview-watch.lock').open('a') as lock:
  fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
  status=out/'checkpoint-preview-watch-status.json'
  errors={}
  while True:
   finished=True
   for ep,variant in TARGETS:
    r=BASE/'runs'/f'episode_{ep:06d}_newton_{variant}'
    if not (r/'flash_chord_coherent_actor_transfer/TRAINING_COMPLETED').exists():finished=False
    reviews=r/'checkpoint-reviews.json'
    if not reviews.exists():continue
    try:
     previous=catalog(ep,Path(json.loads((r/'episode-config.json').read_text())['baseline_run']),include_selected=True)['best']
     previous=attach(previous['run'],previous['selected']) if previous else baseline(ep)
     eligible=[]
     for row in json.loads(reviews.read_text()):
      step=row['environment_steps'];key=f'{ep}:{step}'
      if (out/f'episode_{ep:06d}-checkpoint-{step}-result.json').exists() or errors.get(key,{}).get('attempts',0)>=2:continue
      if not (r/f'flash_chord_coherent_actor_transfer/state_{step}.safetensors').exists():continue
      if better(row,previous):eligible.append(row)
     if eligible:
      row=max(eligible,key=lambda x:x['object_motion_quality']['score']);key=f'{ep}:{row["environment_steps"]}'
      try:publish(ep,row['environment_steps'],variant)
      except Exception:
       errors[key]={'attempts':errors.get(key,{}).get('attempts',0)+1,'error':traceback.format_exc(),'updated_utc':utc()}
       print('Checkpoint preview error',key,errors[key]['error'],flush=True)
    except (OSError,ValueError,KeyError):
     print('Checkpoint preview scan retry',ep,traceback.format_exc(),flush=True)
   save(status,{'updated_utc':utc(),'phase':'completed' if finished else 'watching_full_motion_checkpoints',
    'errors':errors,'partial_episodes_are_never_published':True,'training_completion_is_separate':True})
   if finished:break
   time.sleep(20)
if __name__=='__main__':
 p=argparse.ArgumentParser(description=__doc__);p.add_argument('episode',type=int,nargs='?');p.add_argument('step',type=int,nargs='?')
 p.add_argument('--watch',action='store_true')
 p.add_argument('--variant',default='backward_grasp_20261020',choices=['backward_grasp_20261020','grasp_curriculum_20261020']);a=p.parse_args()
 if a.watch:watch()
 else:
  if a.episode is None or a.step is None:p.error('episode and step are required without --watch')
  publish(a.episode,a.step,a.variant)
