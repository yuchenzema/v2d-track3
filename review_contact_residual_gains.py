"""Bounded, declared action-gain experiments, with continuous unassisted physics."""
import json
from improvement_common import BASE,save,utc
from batch_review import review
from assess_human_events import assess

for ep in [10,16]:
 r=BASE/f'runs/episode_{ep:06d}_newton_grasp_curriculum_20261020'
 for gain in [.5,.75]:
  step=355000320
  checkpoint=('/episode/research/handover-schedule-checkpoint-355000320/policy_355000320.safetensors' if ep==16 else '/episode/flash_chord_coherent_actor_transfer/policy_355000320.safetensors')
  row=review(ep,step,root_override=r,container_override=f'v2d-grasp-episode{ep}-newton',checkpoint_override=checkpoint,
   residual_profile={'wrist_position':gain,'wrist_orientation':1.,'finger':1.})
  label=f'residual-position-{gain}-checkpoint-{step}'
  save(r/f'research/{label}-review.json',row)
  print(ep,gain,{k:row[k] for k in ['source_mean_object_iou','source_worst_phase_object_iou','completion_rate_32_worlds']},flush=True)
