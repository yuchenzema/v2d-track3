---
name: v2d-container
description: Guide a V2D Challenge team through building, checking, and saving the submission container that reproduces its best or selected leaderboard submissions, with policy weights (Tracks 2 and 3) or final reconstructions (Track 1), matching code commit, and matching evaluator and kit hashes.
---

# Build a V2D Challenge submission container

Read [CONTAINER_SUBMISSION.md](../../CONTAINER_SUBMISSION.md) first.
It defines the container layout, the `submission.json` schema, and the checks the organizers run.
Run commands from the kit root, containing `tools/`, `submit/`, and `examples/`.

Tell the participant at the start:
the container is submitted alongside the technical report, leaderboard results stay provisional until the organizers' container checks pass, and mismatches between the container's code, hashes, or results and the leaderboard can delay or invalidate prizes.

The container must hold the exact artifacts behind the counted submissions.
Never edit a submitted file's provenance columns, re-pack a counted submission with a different kit, or substitute a retrained or re-exported checkpoint to make a hash match.
Any of these hides a mismatch the organizers will still find, because they compare against the file Kaggle scored.
When an artifact cannot be found, stop and have the participant contact the organizers.

## 1. Identify the counted submissions

Ask which tracks (and Track 2 tiers) the team entered.
For each Kaggle competition of each entered track, find the submission that counts: the selected Final Submission, or the best-scoring submission if none is selected.
List them with the participant's Kaggle credentials:

```bash
kaggle competitions submissions COMPETITION_SLUG --format json
```

`config/tracks.json` maps each track to its competition slugs.
Group the counted submissions by file.
Each track sends one file to all of its competitions, so expect one file per track, and one per Track 2 tier.
If the counted submissions on one track's boards are different files, each one needs its own entry in `submission.json`, with its own label.

Record each counted submission's ID, file name, and description.
Read the recorded hashes from the file itself in the next step, not from memory or notes.

## 2. Collect the exact submitted files

Find the files that were uploaded, using the upload receipts written by `submit/v2d_submit.py` (`*.kaggle-receipt.json`), which record the uploaded path and submission IDs.
Copy each one to `submitted/` with its receipt.
Print the provenance it carries, which the container must match:

```bash
python3 -c "
import sys, pandas as pd
f = sys.argv[1]
d = pd.read_parquet(f) if f.endswith('.parquet') else pd.read_csv(f, dtype=str)
for c in ['eval_script_sha256','eval_code_sha256','packer_script_sha256','uploader_script_sha256','code_commit_url']:
    if c in d: print(c, d[c].unique())
if 'checkpoint_sha256' in d: print('checkpoints', d['checkpoint_sha256'].nunique())
" submitted/FILE
```

If the local copy may differ from what Kaggle scored, the `--kaggle` check in step 6 compares them.

## 3. Gather the artifacts and match every hash

Work through the hash table in CONTAINER_SUBMISSION.md for each counted file.
Compute each hash from the candidate artifact with `sha256sum` and compare it with the file's column before building anything.

- **Code.** Clone the repository into `code/` and check out the commit in `code_commit_url`.
Keep `.git`.
`git status --porcelain --untracked-files=no` must print nothing.
Confirm the repository is shared with [Nvidia-V2D-Challenge](https://github.com/Nvidia-V2D-Challenge).
- **Kit.** Find the kit copy whose `tools/pack_submission.py` and `submit/v2d_submit.py` hash to `packer_script_sha256` and `uploader_script_sha256`.
Check the participant's working copy, older kit downloads, and the image they packed in.
Put it at `kit/`.
Put this kit version at `verifier_kit/`.
- **Evaluator (Tracks 2 and 3).** Find the `evaluate_policy.py` whose sha256 is `eval_script_sha256`.
The kit copy at `flash_chord/scripts/evaluate_policy.py` is the usual one.
Compute `eval_code_sha256` for the runtime they evaluated in:

```bash
python3 -c "
import hashlib, sys
from pathlib import Path
p = Path(sys.argv[1]); h = hashlib.sha256()
for n in sorted(['__init__.py','harness.py','metrics.py','rollout.py']):
    h.update(n.encode()); h.update(b'\0'); h.update((p/n).read_bytes())
print(h.hexdigest())
" /path/to/flash_chord/src/flash_chord/evaluation
```

- **Checkpoints (Tracks 2 and 3).** For each scored episode, find the checkpoint whose sha256 equals that episode's `checkpoint_sha256`.
Copy it unchanged to `weights/<label>/`.
- **Outputs.** Track 1: copy the final `episode_XXXXXX.npz` files and meshes that were packed to `outputs/<label>/reconstructions/` (required).
Tracks 2 and 3: copy the evaluator rollouts that were packed to `outputs/<label>/rollouts/` (recommended).
Track 3: copy the meshes to `meshes/<label>/`.
Track 2 Tier 3: copy the `--frame-references` input.
- **Track 1 pipeline.** Collect every model the reconstruction pipeline loads, so it runs offline.

Report every hash that does not match, with both values.
A mismatch means the artifact is not the one that produced the counted submission.
Keep searching (other checkpoints from the same training run, other kit copies, the evaluation image), and stop with a clear report when none matches.

## 4. Write `submission.json` and `run.sh`

Start from `examples/container/submission.track2.json` or `examples/container/submission.track1.json`.
Fill in one entry per counted file, with every path relative to `/submission`.
`evaluator.evaluation_package` is the path inside the image.

Copy `examples/container/run.sh` to `run.sh` and set its settings block for the image: `KIT_PYTHON`, `EVAL_PYTHON`, and `RUNTIME_ROOT`.
For Track 1, implement `reconstruct_episodes()` so it runs the participant's pipeline on the listed episodes from the videos under `$DATASET_ROOT`.
Write a short `README.md` with GPU and memory needs and the expected run time.

## 5. Build the image

Lay out a build directory with `submission/` holding everything above, and use `examples/container/Dockerfile`.
Set `BASE_IMAGE` to the image the participant evaluated (Tracks 2 and 3) or reconstructed (Track 1) in.
The image needs `git` and the kit's Python requirements.

```bash
docker build --build-arg BASE_IMAGE=EVAL_IMAGE -t TEAM-v2d-submission:final .
```

## 6. Verify and reproduce

Run every check, without network access, and fix the container until all pass:

```bash
docker run --rm --network none -v DATASET:/data:ro TEAM-v2d-submission:final \
    /submission/run.sh verify --dataset-root /data
```

Then reproduce each counted submission and compare:

```bash
docker run --rm --gpus all --network none -v DATASET:/data:ro -v "$PWD/out:/output" TEAM-v2d-submission:final \
    /submission/run.sh reproduce --label LABEL --dataset-root /data --output /output
docker run --rm --network none -v "$PWD/out:/output" TEAM-v2d-submission:final \
    /submission/run.sh verify --reproduced LABEL=/output/LABEL/FILE_NAME
```

The reproduced file must have the same rows and identical provenance.
Poses and scores differ a little between runs, because the simulator is not bitwise deterministic.
Compare the scores the packer prints with the leaderboard and report the differences.
A large gap needs an explanation before submitting.
For Track 1, run `reproduce --episodes` on two or three sequences and compare with the stored reconstructions.

If the participant's Kaggle token is available, add `--kaggle` to `verify` (with network access) to compare each stored file with Kaggle's copy.

## 7. Save and report

```bash
docker save TEAM-v2d-submission:final | zstd -T0 > TEAM-v2d-submission.tar.zst
sha256sum TEAM-v2d-submission.tar.zst
```

Report to the participant:
the counted submissions and their IDs, the result of every verify check, the reproduced scores next to the leaderboard scores, the image digest and tarball checksum, and anything unresolved.
Remind them to upload the tarball to Google Drive or another file sharing service, and to submit the link, the tarball checksum, and the image digest with the technical report.
The organizers must be able to download it from the link without requesting access.
