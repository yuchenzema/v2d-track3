import json,shutil
from pathlib import Path
from improvement_common import BASE
from batch_native import docker
R=BASE/'runs/episode_000010_newton_continuous_shell_probe_20261020'
shutil.copy2(BASE/'benchmark_cup_contact_scene.py',R/'scripts/benchmark_cup_contact_scene.py')
for count in [512,1024]:
 available=int(next(s.split()[1] for s in Path('/proc/meminfo').read_text().splitlines() if s.startswith('MemAvailable:')))*1024
 if count==1024 and available<25*1024**3:print('1024-world benchmark skipped: preserve RAM reserve',flush=True);continue
 label=f'continuous-contact-benchmark-{count}'
 docker(['exec','v2d-shell-probe-episode10-newton','env',f'V2D_BENCHMARK_WORLDS={count}','V2D_BENCHMARK_REFERENCE=track3_axis_aligned_nesting_probe',f'V2D_BENCHMARK_LABEL={label}','python','-u','/episode/scripts/benchmark_cup_contact_scene.py'],R/f'logs/{label}.log')
 print(count,json.loads((R/f'evaluation/{label}.json').read_text()),flush=True)
