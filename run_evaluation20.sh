#!/usr/bin/env bash
set -euo pipefail
task_dir="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
exec "$task_dir/runs/episode_000005_newton_20261015/.venv-video/bin/python" -u "$task_dir/evaluation20_terminal.py" "$@"
