"""Test declared smaller left-hand actor changes without altering reference or physics."""
import hashlib,json
from improvement_common import BASE,save,baseline,better,verify_preserved
from batch_review import review
R=BASE/'runs/episode_000010_newton_grasp_curriculum_20261020'
frozen=json.loads((R/'coherent-input-checksums.json').read_text())
for weight in [.125,.25,.375]:
 for name,digest in frozen.items():assert hashlib.sha256((R/name).read_bytes()).hexdigest()==digest,name
 row=review(10,350_007_296,root_override=R,container_override='v2d-grasp-episode10-newton',
  checkpoint_override='/episode/research/initial-policy/policy_350007296.safetensors',
  reference_override='track3_collision_variant',
  actor_combination={'alternate_checkpoint':'/episode/research/precision-schedule-checkpoint-360000000/policy_360000000.safetensors',
   'alternate_checkpoint_sha256':hashlib.sha256((R/'research/precision-schedule-checkpoint-360000000/policy_360000000.safetensors').read_bytes()).hexdigest(),
   'hand_weights':{'left':weight,'right':0.}})
 row['passes_baseline_improvement_gate']=better(row,baseline(10))
 save(R/f'research/small-left-mixture-{weight}-review.json',row)
 print(weight,{k:row[k] for k in ['source_mean_object_iou','completion_rate_32_worlds','passes_baseline_improvement_gate']},
  row['object_motion_quality']['objects'],flush=True)
 verify_preserved(10)
