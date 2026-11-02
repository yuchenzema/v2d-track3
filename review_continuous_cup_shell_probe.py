from pathlib import Path
import json
from batch_native import docker
from batch_review import review
from improvement_common import BASE,save,baseline,better
R=BASE/'runs/episode_000010_newton_continuous_shell_probe_20261020'
for source,label in [(160,'continuous-shell-terminal-rest'),(100,'continuous-shell-held100-rest')]:
 docker(['exec','v2d-shell-probe-episode10-newton','env','V2D_DIAGNOSTIC_REFERENCE=track3_hollow_supported_candidate',f'V2D_DIAGNOSTIC_SOURCE_FRAME={source}',f'V2D_DIAGNOSTIC_LABEL={label}','python','-u','/episode/scripts/diagnose_terminal_support.py'],R/f'logs/{label}.log')
 print(label,json.loads((R/f'evaluation/{label}.json').read_text()),flush=True)
row=review(10,360000000,root_override=R,container_override='v2d-shell-probe-episode10-newton',checkpoint_override='/episode/research/initial-policy/policy_360000000.safetensors',reference_override='track3_hollow_supported_candidate',substeps_override=20)
save(R/'research/continuous-shell-seed-actor-review.json',row)
print('Seed actor review',row['source_mean_object_iou'],row['completion_rate_32_worlds'],better(row,baseline(10)),flush=True)
