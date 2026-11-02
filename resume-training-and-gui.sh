#!/usr/bin/env bash
set -euo pipefail

cd -- "$(dirname -- "${BASH_SOURCE[0]}")"
project_root=$PWD
checkpoint="$project_root/runs/episode_000030_newton_pickup_repair_20261026/flash_chord_coherent_actor_transfer/state_30000128.safetensors"
test -s "$checkpoint"
test -s "${checkpoint/state_/policy_}"

# Check host resources before changing the saved pause state.
nvidia-smi --query-gpu=name,memory.free --format=csv,noheader
sg docker -c 'docker info --format "Docker server: {{.ServerVersion}}"'
systemctl --user show-environment >/dev/null

systemctl --user enable --now v2d-track3-gui.service v2d-episode30-pickup-repair.service
systemctl --user is-active --quiet v2d-track3-gui.service
systemctl --user is-active --quiet v2d-episode30-pickup-repair.service

.venv/bin/python - <<'PY'
import json
import time
import urllib.request
from datetime import datetime, timezone
from pathlib import Path

for attempt in range(30):
    try:
        with urllib.request.urlopen('http://127.0.0.1:8768/api/status', timeout=5) as response:
            if response.status != 200:
                raise RuntimeError('Dashboard returned an unexpected status')
        break
    except OSError:
        if attempt == 29:
            raise RuntimeError('Dashboard is not ready; pause status was preserved')
        time.sleep(1)

path = Path('runs/evaluation_batch_20261015/user-pause.json')
state = json.loads(path.read_text())
state.update(
    status='resume_requested',
    updated_utc=datetime.now(timezone.utc).isoformat(),
    reason='User requested resuming saved brush pickup training and GUI.',
    automatic_resume_allowed=True,
    enabled_services=['v2d-track3-gui', 'v2d-episode30-pickup-repair'],
    disabled_services=[name for name in state.get('disabled_services', [])
                       if name != 'v2d-episode30-pickup-repair'],
    training_progress_verified=False,
)
backup = path.with_name('user-pause.before-resume.json')
if not backup.exists():
    backup.write_text(path.read_text())
temporary = path.with_suffix('.resume.tmp')
temporary.write_text(json.dumps(state, indent=2) + '\n')
temporary.replace(path)
PY

if command -v xdg-open >/dev/null; then
  xdg-open http://127.0.0.1:8768/ || true
fi
printf 'Dashboard: http://127.0.0.1:8768/\n'
printf 'Resume service started; confirm checkpoint advancement in the training log:\n'
printf '%s\n' "$project_root/runs/episode_000030_newton_pickup_repair_20261026/logs/newton-training.log"
systemctl --user status v2d-track3-gui.service v2d-episode30-pickup-repair.service --no-pager
