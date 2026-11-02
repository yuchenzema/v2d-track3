"""Review the trained spoon refinements with their original250N/m wrist gains."""
import os,json,traceback
from improvement_common import BASE,BATCH,save,utc,better
from batch_review import review
from video_catalog import catalog
R=BASE/'runs/episode_000032_newton_regularized_spoon_20261021';A=BATCH/'improvements_20261021/activities/episode_000032-restored-training-gains.json'
def status(state,phase,**extra):save(A,{'episode':32,'pid':os.getpid(),'process_script':__file__.split('/')[-1],'updated_utc':utc(),'status':state,'phase':phase,'description':'Rechecking completed spoon-policy training with the250N/m gains actually used during learning. The accepted70.1% overlap video stays current unless a new improvement is verified.',**extra})
def main():
 results=[];profile={side:{'kp':250.,'kd':20.,'angular_kp':25.} for side in ['left','right']};folder=R/'flash_chord_coherent_actor_transfer';steps=sorted(int(p.stem.split('_')[-1]) for p in folder.glob('policy_*.safetensors'))
 for step in sorted(set([steps[-1],steps[0]]),reverse=True):
  status('running',f'Whole-source32-world gain-corrected review of trained checkpoint{step}')
  row=review(32,step,root_override=R,container_override='v2d-regularized-episode32-newton',substeps_override=20,recording_world_count=32,wrist_calibration=profile,tracking_feedback={'left':2.,'right':2.,'maximum_correction_m':.015,'release_lead_source_frames':16})
  accepted=catalog(32,BASE/'runs/episode_000032_newton_batch_20261015',True)['best']['selected'];eligible=better(row,accepted);row['passes_current_accepted_policy_gate']=eligible;path=R/f'research/restored-training-gains-{step}-review.json';save(path,row);results.append({'review':str(path),'eligible':eligible,'mean_iou':row['source_mean_object_iou'],'completion_rate':row['completion_rate_32_worlds']});print('Spoon actual training gains',step,row['source_mean_object_iou'],row['completion_rate_32_worlds'],'better',eligible,flush=True)
 status('completed','Gain-corrected trained spoon reviews complete',eligible_for_independent_publication=any(x['eligible'] for x in results),results=results);save(R/'research/restored-training-gains-completion.json',{'updated_utc':utc(),'results':results})
if __name__=='__main__':
 try:main()
 except Exception:status('failed','Spoon gain-corrected evaluation failed; accepted video retained',error=traceback.format_exc());raise
