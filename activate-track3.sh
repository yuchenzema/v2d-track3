# Source this file: source ~/v2d-track3/activate-track3.sh [reconstruction|groot|submission]
V2D_TRACK3_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
export V2D_TRACK3_ROOT
case "${1:-reconstruction}" in
    reconstruction) source "$V2D_TRACK3_ROOT/.venv/bin/activate" ;;
    groot) source "$V2D_TRACK3_ROOT/Isaac-GR00T/.venv/bin/activate" ;;
    submission) source "$V2D_TRACK3_ROOT/.venv-submission/bin/activate" ;;
    *) printf '%s\n' 'Choose reconstruction, groot, or submission.' >&2; return 2 ;;
esac
export PATH="$V2D_TRACK3_ROOT/../.local/bin:$PATH"
export PYTHON="$VIRTUAL_ENV/bin/python"
export V2D_DATASET_ROOT="$V2D_TRACK3_ROOT/datasets/video_to_data_challenge"
export ISAAC_GROOT_DIR="$V2D_TRACK3_ROOT/Isaac-GR00T"
export MANO_DIR="$V2D_TRACK3_ROOT/models/mano"
printf 'Environment: %s\nDataset: %s\n' "$VIRTUAL_ENV" "$V2D_DATASET_ROOT"
