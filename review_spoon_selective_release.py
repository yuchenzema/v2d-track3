"""Compare source-observed finger releases on a preserved trained spoon actor."""
import json
from improvement_common import BASE,save,baseline,better,verify_preserved
from batch_review import review
R=BASE/'runs/episode_000016_newton_stationary_hold_20261020';STEP=370000384
base={'side':'right','open_posture_source_frame':140,'finger_openness_scale':0.}
nonpinch={**base,'finger_joint_prefixes':['right_middle_','right_ring_','right_pinky_'],'after_source_frame':90,'ramp_source_frames':10}
for label,spec in [
 ('full-release-131',[{**base,'after_source_frame':131,'ramp_source_frames':4}]),
 ('selective-release-131',[nonpinch,{**base,'after_source_frame':131,'ramp_source_frames':4}]),
 ('selective-release-126',[nonpinch,{**base,'after_source_frame':126,'ramp_source_frames':4}])]:
 row=review(16,STEP,root_override=R,container_override='v2d-stationary-episode16-newton',
  checkpoint_override='/episode/research/review-only-stationary-5M/policy_370000384.safetensors',
  reference_override='track3_stationary_spoon_hold_probe',substeps_override=20,release_spec=spec)
 row['passes_baseline_improvement_gate']=better(row,baseline(16));save(R/f'research/{label}-review.json',row)
 print(label,{k:row[k] for k in ['source_mean_object_iou','completion_rate_32_worlds','passes_baseline_improvement_gate']},flush=True)
 verify_preserved(16)
