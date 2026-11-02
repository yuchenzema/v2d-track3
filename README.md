# V2D Track 3 evaluation work

Source snapshot for egocentric reconstruction, source-frame CAD alignment,
robot reference construction, residual policy training, and recorded evaluation.

## Current status

As of 2026-11-02, Kaggle has administratively invalidated all 29 previous
submissions across six competitions. There are currently no valid scores and
strict episode completion is **0/20**. The API gives no more specific reason.

The official FAQ requires own reconstructed meshes to generate the policy
trajectories, even when public scans exist. The historical geometry, texture,
marker-template and actor lineage used public scans and is excluded from the
new evaluation-only run. Original artifacts and receipts remain archived.
The current serial controller rebuilds geometry from evaluation RGB and metric
depth, rechecks every source frame, and trains fresh policies. Historical EP38
training (1,280,000 steps per trial) is not compliant new-run progress.
See `submission_docs/own-geometry-correction-20261102.txt`.

The evaluation roster is:
`1,3,5,6,8,10,14,15,16,19,20,25,28,30,32,33,35,36,37,38`.

## Repository contents

- Top-level Python files: reconstruction, fitting, training, evaluation,
  export, monitoring, and focused regression tests.
- `run_evaluation20.sh`: the existing serial controller's local monitor.
  GUI refreshes every 5 seconds; terminal progress is printed every 10 minutes.
- `package_existing_evaluation20.py`: archive selected existing artifacts,
  retaining original provenance and quality failures. It does not upload them.
- `v2d_submission_kit/`: the earlier local kit used by legacy export adapters.
- `submission_kit_20261102/`: exact contents of the participant-provided kit ZIP,
  without modifying the official evaluator, metrics, packer or uploader.
- `submission_docs/`: recorded dependencies, local changes and source hashes.

Datasets, checkpoints, videos, run directories, credentials and the multi-GB
delivery archive are excluded from Git. No Kaggle token belongs in this repo.

## Environment and local execution

The current orchestration targets Linux, a working NVIDIA Docker runtime, and
an RTX 5090. Host Python environments and Docker images are maintained separately.
Existing scripts use `/home/yu_eli/v2d-track3` and per-run paths; this snapshot
is not an independently verified portable/offline container.

Do not start a second episode worker. The active pipeline uses
`v2d-track3-focused.service`, the global episode execution lock, and
`v2d-track3-gui.service` at `http://localhost:8768`.

On the original machine:

```bash
./run_evaluation20.sh status
./run_evaluation20.sh
```

Controller changes are loaded at a safe episode boundary by the existing
health service. A failed quality gate remains a failure; an overlay inspection
is never counted as a passed episode.

## Formal submission requirements

The original scan-derived cohort is blocked from new uploads. Historical IDs
can still be queried without uploading:

```bash
./submit_original20.sh --poll
```

`own_submission_queue.py` runs inside the same serial controller. Once all 20
own-geometry policies pass the existing local motion/contact/grasp checks and
official rollout provenance checks, it packs and validates the cohort, pushes
the producing commit and submits to all six metrics. No extra approval or new
worker is required. Existing receipts prevent duplicate uploads. Exact-match
strict completion remains separate. `refresh_kaggle_status.py` refreshes current
API states every ten minutes, overriding invalid historical scores in the GUI.

Read `submission_kit_20261102/tracks/track_3/README.md` and
`submission_kit_20261102/CONTAINER_SUBMISSION.md`.

Track 3 requires one policy per evaluation episode and reconstructed meshes for
all 36 episode/object pairs. The combined submission has 9,120 pose rows.
Use the official evaluator at its required runtime revision, then the unmodified
packer to combine actual rollout trajectories and meshes into one Parquet.

The producing code must be committed **before** evaluation. This new commit
cannot retroactively identify the implementation behind old recordings.
Legacy staging hashes identify their export adapter, not the official evaluator;
do not rename or replace these hashes to claim a new evaluation.

After official evaluation:

```bash
python3 submission_kit_20261102/tools/pack_submission.py \
  --track track_3 --results /path/to/official-results \
  --meshes /path/to/selected-meshes --code-commit "$COMMIT_URL" \
  -o track3_submission.parquet
python3 submission_kit_20261102/tools/preflight.py \
  --track track_3 --results /path/to/official-results
python3 submission_kit_20261102/submit/v2d_submit.py \
  track3_submission.parquet --track track_3 \
  -m "Track 3 evaluation" --dry-run
```

Kaggle rule acceptance is required on all six metric competitions. Upload only
after the entry, provenance, mesh and format checks pass; retain exact uploaded
files and receipts. An accepted upload is not a successful imitation score.
The final technical report must accompany an offline reproducible container,
its downloadable image archive, SHA256 and image digest. The container's
`verify` and `reproduce` checks must pass against the exact counted submissions.
