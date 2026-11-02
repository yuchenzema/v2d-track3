"""Evaluate the trained episode30 model with its actual training wrist gains."""
import os,json,traceback
from improvement_common import BASE,BATCH,save,utc,better
from batch_review import review
from video_catalog import catalog
R=BASE/'runs/episode_000030_newton_consistent_brush_20261021';A=BATCH/'improvements_20261021/activities/episode_000030-restored-training-gains.json'
def status(state,phase,**extra):save(A,{'episode':30,'pid':os.getpid(),'process_script':__file__.split('/')[-1],'updated_utc':utc(),'status':state,'phase':phase,'description':'Evaluating the trained model with right-wrist250N/m as used in training. Checkpoint key sorting had silently reduced inference to25N/m. Original geometry and reference are preserved.',**extra})
def main():
 status('running','Whole-source32-world evaluation after restoring training motor-gain precedence')
 results=[];calibration={'left':{'kp':25.,'kd':6.,'angular_kp':25.},'right':{'kp':250.,'kd':20.,'angular_kp':25.}}
 steps=sorted(int(p.stem.split('_')[-1]) for p in (R/'flash_chord_coherent_actor_transfer').glob('policy_*.safetensors'))
 for step in sorted(set([steps[-1],steps[0]]),reverse=True):
  row=review(30,step,root_override=R,container_override='v2d-consistent-episode30-newton',substeps_override=20,recording_world_count=32,wrist_calibration=calibration)
  accepted=catalog(30,BASE/'runs/episode_000030_newton_batch_20261015',True)['best']['selected'];eligible=better(row,accepted);row['passes_current_accepted_policy_gate']=eligible;path=R/f'research/restored-training-gains-{step}-review.json';save(path,row);results.append({'review':str(path),'eligible':eligible,'mean_iou':row['source_mean_object_iou'],'completion_rate':row['completion_rate_32_worlds']});print('Training wrist gains',step,row['source_mean_object_iou'],row['completion_rate_32_worlds'],'better',eligible,flush=True)
 status('completed','Training-gain-corrected whole-source evaluations finished',eligible_for_independent_publication=any(x['eligible'] for x in results),results=results);save(R/'research/restored-training-gains-completion.json',{'updated_utc':utc(),'results':results})
if __name__=='__main__':
 try:main()
 except Exception:status('failed','Training-gain correction evaluation failed; prior video retained',error=traceback.format_exc());raise
