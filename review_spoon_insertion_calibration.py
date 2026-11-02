"""Test bounded robot wrist-command calibration against measured insertion error."""
from improvement_common import BASE,save,baseline,better,verify_preserved
from batch_review import review
R=BASE/'runs/episode_000016_newton_stationary_hold_20261020'
for label,scale,release in [('half-insertion-calibration',.5,None),('full-insertion-calibration',1.,None),
 ('full-insertion-calibration-release',1.,[{'side':'right','after_source_frame':131,'ramp_source_frames':4,'open_posture_source_frame':140,'finger_openness_scale':0.}])]:
 profile={'wrist_position':1.,'wrist_orientation':1.,'finger':1.,'after_source_frame':78,'ramp_source_frames':17,
  'until_source_frame':131,'fade_source_frames':4,'wrist_position_bias_m':{'right':[scale*v for v in [-.016,-.034,-.032]]}}
 row=review(16,370000384,root_override=R,container_override='v2d-stationary-episode16-newton',
  checkpoint_override='/episode/research/review-only-stationary-5M/policy_370000384.safetensors',
  reference_override='track3_stationary_spoon_hold_probe',substeps_override=20,residual_profile=profile,release_spec=release)
 row['passes_baseline_improvement_gate']=better(row,baseline(16));save(R/f'research/{label}-review.json',row)
 print(label,{k:row[k] for k in ['source_mean_object_iou','completion_rate_32_worlds','passes_baseline_improvement_gate']},flush=True);verify_preserved(16)
