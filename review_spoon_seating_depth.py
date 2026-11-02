"""Test measured vertical seating correction on the better10M spoon checkpoint."""
from improvement_common import BASE,save,baseline,better,verify_preserved
from batch_review import review
R=BASE/'runs/episode_000016_newton_stationary_hold_20261020'
release=[{'side':'right','after_source_frame':131,'ramp_source_frames':4,'open_posture_source_frame':140,'finger_openness_scale':0.}]
for label,depth,spec in [('seat40mm',.04,None),('seat40mm-release',.04,release),('seat60mm-release',.06,release)]:
 profile={'wrist_position':1.,'wrist_orientation':1.,'finger':1.,'after_source_frame':78,'ramp_source_frames':22,
  'until_source_frame':131,'fade_source_frames':4,'wrist_position_bias_m':{'right':[0.,0.,-depth]}}
 row=review(16,375000064,root_override=R,container_override='v2d-stationary-episode16-newton',
  checkpoint_override='/episode/research/precision-stage-checkpoint-375000064/policy_375000064.safetensors',
  reference_override='track3_stationary_spoon_hold_probe',substeps_override=20,residual_profile=profile,release_spec=spec)
 row['passes_baseline_improvement_gate']=better(row,baseline(16));path=R/f'research/{label}-review.json';save(path,row)
 print(label,{k:row[k] for k in ['source_mean_object_iou','completion_rate_32_worlds','passes_baseline_improvement_gate']},flush=True);verify_preserved(16)
 if row['passes_baseline_improvement_gate']:
  from publish_contact_controller_candidate import publish
  publish(16,path,'spoon-'+label,root_override=R)
