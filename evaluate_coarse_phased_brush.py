"""Isolate pickup control from the fine-model passive-support regression."""
import os,json,shutil,traceback
import pyarrow as pa,pyarrow.parquet as pq
from improvement_common import BASE,BATCH,save,utc,better
from batch_review import review
from video_catalog import catalog
R=BASE/'runs/episode_000030_newton_consistent_brush_20261021';REF='track3_brush_phased_coarse';A=BATCH/'improvements_20261021/activities/episode_000030-coarse-phased-rollout.json'
def status(state,phase,**extra):save(A,{'episode':30,'pid':os.getpid(),'process_script':__file__.split('/')[-1],'updated_utc':utc(),'status':state,'phase':phase,'description':'Testing the phased robot grasp on the original rigid brush geometry and tested initial support. No compliant material, object guidance or failure resets.',**extra})
def main():
 status('running','Independent native32-world ablation with the original rigid brush model')
 target=R/'human_motion_data'/REF;assert not target.exists();shutil.copytree(R/'human_motion_data/track3_brush_phased_grasp',target);path=next(target.rglob('motion.parquet'));row=pq.ParquetFile(path).read().to_pylist()[0];row['object_urdf_paths'][0]='/episode/reconstruction/mesh_priors/mini_sweeper.urdf';pq.write_table(pa.Table.from_pylist([row]),path,compression='zstd')
 row=review(30,45001728,reference_override=REF,root_override=R,container_override='v2d-consistent-episode30-newton',substeps_override=20,recording_world_count=32,tracking_feedback={'left':2.,'right':.5,'maximum_correction_m':.03,'right_wrist_residual_gain':0.,'right_finger_residual_gain':0.},wrist_calibration={'left':{'kp':25.,'kd':6.,'angular_kp':25.},'right':{'kp':500.,'kd':30.,'angular_kp':100.}})
 accepted=catalog(30,BASE/'runs/episode_000030_newton_batch_20261015',True)['best']['selected'];eligible=better(row,accepted);row['passes_current_accepted_policy_gate']=eligible;save(R/'research/coarse-phased-brush-policy-review.json',row);status('completed','Original-geometry phased rollout reviewed',eligible_for_independent_publication=eligible,source_mean_object_iou=row['source_mean_object_iou'],completion_rate=row['completion_rate_32_worlds']);print('Coarse phased brush',row['source_mean_object_iou'],row['completion_rate_32_worlds'],'better',eligible,flush=True)
if __name__=='__main__':
 try:main()
 except Exception:status('failed','Original-geometry phased rollout failed; prior video retained',error=traceback.format_exc());raise
