"""Test vertical-only wrist impedance over the source-observed blue-cup lift."""
import hashlib
from batch_review import review
from improvement_common import BASE,save,baseline,better,verify_preserved
R=BASE/'runs/episode_000010_newton_grasp_curriculum_20261020';alt=R/'research/precision-schedule-checkpoint-360000000/policy_360000000.safetensors'
actors={'alternate_checkpoint':'/episode/'+str(alt.relative_to(R)),'alternate_checkpoint_sha256':hashlib.sha256(alt.read_bytes()).hexdigest(),'hand_weights':{'left':.5,'right':0.}}
for label,kp,kd,combination in [('vertical100-mixture',100,12,actors),('vertical250-mixture',250,18,actors),('vertical250-preserved',250,18,None)]:
 profile={'left':{'kp':kp,'kd':kd,'start_source_frame':60,'ramp_source_frames':8,'until_source_frame':95,'fade_source_frames':12,'axes':['z']}}
 row=review(10,350007296,root_override=R,container_override='v2d-grasp-episode10-newton',
  checkpoint_override='/episode/research/initial-policy/policy_350007296.safetensors',reference_override='track3_collision_variant',actor_combination=combination,timed_impedance=profile)
 row['passes_baseline_improvement_gate']=better(row,baseline(10));save(R/f'research/{label}-review.json',row)
 print(label,{k:row[k] for k in ['source_mean_object_iou','completion_rate_32_worlds','passes_baseline_improvement_gate']},flush=True);verify_preserved(10)
