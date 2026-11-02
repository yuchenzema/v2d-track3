"""Test robot wrist feedforward during the observed outer-cup lift, preserving timing."""
import hashlib
from batch_review import review
from improvement_common import BASE,save,baseline,better,verify_preserved
R=BASE/'runs/episode_000010_newton_grasp_curriculum_20261020';alternate=R/'research/precision-schedule-checkpoint-360000000/policy_360000000.safetensors'
actors={'alternate_checkpoint':'/episode/'+str(alternate.relative_to(R)),'alternate_checkpoint_sha256':hashlib.sha256(alternate.read_bytes()).hexdigest(),'hand_weights':{'left':.5,'right':0.}}
for height in [.02,.04,.06]:
 profile={'wrist_position':1.,'wrist_orientation':1.,'finger':1.,'after_source_frame':55,'ramp_source_frames':12,
  'until_source_frame':95,'fade_source_frames':12,'wrist_position_bias_m':{'left':[0.,0.,height]}}
 row=review(10,350007296,root_override=R,container_override='v2d-grasp-episode10-newton',
  checkpoint_override='/episode/research/initial-policy/policy_350007296.safetensors',reference_override='track3_collision_variant',
  actor_combination=actors,residual_profile=profile)
 row['passes_baseline_improvement_gate']=better(row,baseline(10));save(R/f'research/left-lift-calibration-{height}-review.json',row)
 print(height,{k:row[k] for k in ['source_mean_object_iou','completion_rate_32_worlds','passes_baseline_improvement_gate']},flush=True);verify_preserved(10)
