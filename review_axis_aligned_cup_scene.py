"""Verify the continuous cavity, corrected insertion, and unchanged body properties."""
import json,shutil,time
from pathlib import Path
import numpy as np
from batch_native import docker,_docker
from batch_review import review
from improvement_common import BASE,save,baseline,better
R=BASE/'runs/episode_000010_newton_continuous_shell_probe_20261020';S=BASE/'runs/episode_000010_newton_stable_contact_20261020';ref='track3_axis_aligned_nesting_probe'
while not (R/'research/axis-aligned-nesting-fk.json').exists():time.sleep(5)
_docker(['exec','v2d-shell-probe-episode10-newton','env','CUDA_VISIBLE_DEVICES=','JAX_PLATFORMS=cpu',f'V2D_AUDIT_REFERENCE={ref}',
 'python','-u','/episode/scripts/run_geometry_cpu.py','/episode/scripts/native_reference_audit.py'],R/'logs/axis-aligned-nesting-collision-audit.log')
report=json.loads((R/f'evaluation/{ref}-collision-audit.json').read_text());assert report['source_frames_over_10mm']==0
for source,label in [(160,'axis-aligned-terminal-rest'),(100,'axis-aligned-held100-rest')]:
 docker(['exec','v2d-shell-probe-episode10-newton','env',f'V2D_DIAGNOSTIC_REFERENCE={ref}',f'V2D_DIAGNOSTIC_SOURCE_FRAME={source}',f'V2D_DIAGNOSTIC_LABEL={label}',
  'python','-u','/episode/scripts/diagnose_terminal_support.py'],R/f'logs/{label}.log')
 a=json.loads((R/f'evaluation/{label}.json').read_text());assert a['frozen_source_frame']==source and a['all_states_finite']
 print(label,a['objects'],flush=True)
for root,container,reference,label in [(S,'v2d-stable-episode10-newton','track3_hollow_supported_candidate','four-band-contact-benchmark'),
 (R,'v2d-shell-probe-episode10-newton',ref,'continuous-contact-benchmark')]:
 shutil.copy2(BASE/'benchmark_cup_contact_scene.py',root/'scripts/benchmark_cup_contact_scene.py')
 docker(['exec',container,'env',f'V2D_BENCHMARK_REFERENCE={reference}',f'V2D_BENCHMARK_LABEL={label}',
  'python','-u','/episode/scripts/benchmark_cup_contact_scene.py'],root/f'logs/{label}.log')
 print(label,json.loads((root/f'evaluation/{label}.json').read_text()),flush=True)
with np.load(S/'evaluation/four-band-contact-benchmark-body-properties.npz') as a,np.load(R/'evaluation/continuous-contact-benchmark-body-properties.npz') as b:
 for name in a.files:np.testing.assert_array_equal(a[name],b[name],err_msg=name)
print('Native body names, masses and inertia tensors identical.',flush=True)
row=review(10,360000000,root_override=R,container_override='v2d-shell-probe-episode10-newton',checkpoint_override='/episode/research/initial-policy/policy_360000000.safetensors',reference_override=ref,substeps_override=20)
save(R/'research/axis-aligned-seed-actor-review.json',row)
print('Aligned seed actor',row['source_mean_object_iou'],row['completion_rate_32_worlds'],better(row,baseline(10)),flush=True)
