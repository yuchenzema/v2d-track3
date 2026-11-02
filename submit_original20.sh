#!/usr/bin/env bash
set -euo pipefail
V2D_SUBMISSION_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
exec "$V2D_SUBMISSION_DIR/.venv-submission/bin/python" "$V2D_SUBMISSION_DIR/submit_original20.py" "$@"
