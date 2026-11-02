"""Check saved completed training branches after fixing motor-override precedence."""
import os,json,traceback
from improvement_common import BASE,BATCH,save,utc,better
from batch_review import review
from video_catalog import catalog
A=BATCH/'improvements_20261021/activities/episode_000032-saved-gain-reviews.json'
def status(state,phase,**extra):save(A,{'episode':32,'pid':os.getpid(),'process_script':__file__.split('/')[-1],'updated_utc':utc(),'status':state,'phase':phase,'description':'Rechecking saved spoon-policy training branches with their intended250N/m wrists and the accepted feedback controller. Current video remains protected by full-source improvement checks.',**extra})
def main():
 results=[]
 for variant in ['conservative_spoon_20261021','contact_precision_full_20261021','contact_precision_20261021']:
  root=BASE/f'runs/episode_000032_newton_{variant}';cfg=json.loads((root/'episode-config.json').read_text());folder=root/'flash_chord_coherent_actor_transfer';steps=sorted(int(p.stem.split('_')[-1]) for p in folder.glob('policy_*.safetensors'))
  if not steps:continue
  selected_path=root/'final-controller-selection.json';chosen=json.loads(selected_path.read_text())['selected']['environment_steps'] if selected_path.exists() else steps[-1]
  candidates=sorted({steps[-1],chosen}&set(steps),reverse=True)
  for step in candidates:
   status('running',f'Gain-corrected native32-world review: {variant}, checkpoint{step}')
   profile={side:{'kp':250.,'kd':20.,'angular_kp':25.} for side in ['left','right']}
   row=review(32,step,root_override=root,container_override=cfg['native_container'],substeps_override=20,recording_world_count=32,wrist_calibration=profile,tracking_feedback={'left':2.,'right':2.,'maximum_correction_m':.015,'release_lead_source_frames':16})
   accepted=catalog(32,BASE/'runs/episode_000032_newton_batch_20261015',True)['best']['selected'];eligible=better(row,accepted);row['passes_current_accepted_policy_gate']=eligible;path=root/f'research/restored-training-gains-feedback-{step}-review.json';save(path,row);results.append({'run':str(root),'review':str(path),'eligible':eligible,'mean_iou':row['source_mean_object_iou'],'completion_rate':row['completion_rate_32_worlds']});print(variant,step,row['source_mean_object_iou'],row['completion_rate_32_worlds'],'better',eligible,flush=True)
 save(BATCH/'improvements_20261021/saved-spoon-gain-review-completion.json',{'updated_utc':utc(),'results':results});status('completed','Saved trained spoon branches reviewed with corrected motor precedence',eligible_for_independent_publication=any(r['eligible'] for r in results),results=results)
if __name__=='__main__':
 try:main()
 except Exception:status('failed','Saved-gain review failed; accepted video retained',error=traceback.format_exc());raise
