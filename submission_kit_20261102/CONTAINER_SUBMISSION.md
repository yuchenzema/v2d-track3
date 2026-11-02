# Submission containers

Leaderboard positions are provisional.
After the Competition closes, every team that wants to be considered for an award submits a container alongside its technical report.
The organizers run each container to confirm that it reproduces the team's best or selected leaderboard submissions, with the same code, the same evaluator, and the same model weights.
All results remain provisional until these container checks pass.
**Mismatches between the container's code, hashes, or results and those reported on the leaderboard can delay or invalidate prizes.**

This guide describes what the container must hold, how to check it before you send it, and what the organizers check.
For a guided build, point your agent at [skills/v2d-container/SKILL.md](skills/v2d-container/SKILL.md).

## Which submissions the container must cover

For every Kaggle competition in each track you entered, the container must reproduce the submission that counts for that competition.
This is your selected Final Submission, or, if you selected none, your best-scoring submission on that leaderboard.
Because each track sends one file to all of its metric competitions, this is normally one file per track, and one file per tier in Track 2.
If the submissions that count on different boards of one track are different files, the container must cover every one of them.

List the counted submissions before you start, with their Kaggle submission IDs:

```bash
kaggle competitions submissions v2d-challenge-track2-tier1-auc
```

## The hashes must match the leaderboard

Every Track 2 and Track 3 submission records the hashes of the evaluator, the kit scripts, and each episode's policy.
**The evaluation and submission script hashes in the container must match the hashes recorded in your best or selected Kaggle submissions.**
The organizers recompute each hash from the container and compare it with the file Kaggle holds.

| Column | Hash of | Must match |
|---|---|---|
| `eval_script_sha256` | The `evaluate_policy.py` that rolled out your policies | The evaluator script in the container |
| `eval_code_sha256` | The FlashCHORD evaluation package (`flash_chord/evaluation/`: `__init__.py`, `harness.py`, `metrics.py`, `rollout.py`) | The runtime installed in the container |
| `checkpoint_sha256` | Each episode's policy checkpoint, byte for byte | The checkpoint for that episode in the container |
| `packer_script_sha256` | The kit's `tools/pack_submission.py` | The packer in the container's kit copy |
| `uploader_script_sha256` | The kit's `submit/v2d_submit.py` | The uploader in the container's kit copy |
| `code_commit_url` | The commit of your code | The commit checked out in the container, with no modified files |

Track 1 submissions record only `code_commit_url`, and the same commit rule applies.

These hashes match only if the container holds the exact artifacts that produced the counted submissions.
In practice:

- Keep the kit copy that packed and uploaded each counted submission.
If you updated the kit after that submission, ship the older copy for it.
- Keep the FlashCHORD runtime you evaluated with.
Reinstalling or upgrading it can change `eval_code_sha256`.
- Ship the checkpoint files that were evaluated, unchanged.
Re-saving, converting, or re-exporting a checkpoint changes its hash, even when the weights are equal.
- Commit every change before you evaluate, and record that commit.
Uncommitted edits at evaluation time mean the commit does not reflect the code that ran.

Never edit the provenance columns of a submitted file, and never re-pack a counted submission with a different kit to make hashes line up.
The organizers compare against the file Kaggle scored.
If an artifact behind a counted submission is lost, tell the organizers instead of substituting another.

## What every container holds

The container is a Docker image with everything under `/submission`:

| Path | Contents |
|---|---|
| `submission.json` | What each counted submission was made from (schema below). |
| `run.sh` | The entry point. Start from [examples/container/run.sh](examples/container/run.sh). |
| `code/` | Your repository at the commit in `code_commit_url`, with its `.git` directory, and no modified tracked files. |
| `kit/` | The submission kit copy that packed and uploaded your counted submissions. |
| `verifier_kit/` | This kit version, which provides `tools/verify_container.py`. |
| `submitted/` | The exact files you uploaded to Kaggle for the counted submissions, and their upload receipts. |
| `README.md` | Hardware needs, run time, and anything else a reviewer needs to run the container. |

The image must have `git` and the kit's Python requirements installed.
It must run without network access, with the challenge dataset mounted read-only at `/data` and results written to `/output`.

### Tracks 2 and 3: policy weights and evaluation

| Path | Contents |
|---|---|
| `weights/<label>/episode_XXXXXX.<ext>` | One policy checkpoint per scored episode, the exact files that were evaluated. |
| The FlashCHORD runtime | Installed in the image as you evaluated with it, at the [pinned evaluator commit](README.md#policy-evaluation-for-tracks-2-and-3). |
| `outputs/<label>/rollouts/` | Recommended: the evaluator outputs (`episode_XXXXXX.parquet` and `.json`) that were packed into the counted submission. |
| `meshes/<label>/` | Track 3: the reconstructed meshes packed with the trajectories, named as in the [Track 3 guide](tracks/track_3/README.md). |
| Track 2 Tier 3 frame references | The reconstructed input passed to the packer with `--frame-references`. |

`run.sh reproduce` must roll out every episode's policy with the evaluator and pack a submission file, using one command.
Build the image from the image you evaluated in, so the runtime that produced the leaderboard files is the one in the container.

### Track 1: code and final reconstructions

| Path | Contents |
|---|---|
| `outputs/<label>/reconstructions/` | The final per-episode `episode_XXXXXX.npz` files and meshes packed into the counted submission. |
| Model weights | Every model your pipeline loads, so it runs without downloads. |

`run.sh pack` must recreate the submitted Parquet from the stored reconstructions.
The verifier re-packs them and compares the result with the submitted file.
`run.sh reproduce --episodes LIST` must run your reconstruction pipeline on the listed sequences from the released videos.
The organizers may run it on a few sample sequences and compare the output with your stored reconstructions.

## `submission.json`

One entry per counted submission.
Relative paths are relative to `/submission`.
[examples/container/submission.track2.json](examples/container/submission.track2.json) and [examples/container/submission.track1.json](examples/container/submission.track1.json) are complete templates.

```json
{
  "schema": "v2d_container_v1",
  "team": "Kaggle team name",
  "code": {"path": "code", "commit_url": "https://github.com/ORG/REPO/commit/<full sha>"},
  "kit": "kit",
  "submissions": [
    {
      "label": "tier1",
      "track": "track_2_tier1",
      "file": "submitted/track2_tier1.csv",
      "kaggle_submission_ids": {"add_auc": 57000001, "spider_sr": 57000002, "maniptrans_sr": 57000003, "mppe_cm": 57000004},
      "evaluator": {"script": "kit/flash_chord/scripts/evaluate_policy.py",
                    "evaluation_package": "/path/to/flash_chord/src/flash_chord/evaluation"},
      "checkpoints": {"0": "weights/tier1/episode_000000.safetensors", "1": "weights/tier1/episode_000001.safetensors"},
      "outputs": "outputs/tier1/rollouts"
    }
  ]
}
```

| Field | Meaning |
|---|---|
| `label` | Your name for the submission, used by `run.sh --label` and in the output directory. |
| `track` | The kit's track value: `track_1`, `track_2_tier1`, `track_2_tier2`, `track_2_tier3`, or `track_3`. |
| `file` | The file you uploaded to Kaggle. |
| `kaggle_submission_ids` | The Kaggle submission ID on each metric competition, keyed as in `config/tracks.json`. |
| `kit` | The kit copy that packed and uploaded this submission (defaults to the top-level `kit`). |
| `evaluator.script` | Tracks 2 and 3: the `evaluate_policy.py` that ran. |
| `evaluator.evaluation_package` | Tracks 2 and 3: the `flash_chord/evaluation` directory of the runtime that ran. |
| `checkpoints` | Tracks 2 and 3: episode index to checkpoint file. |
| `outputs` | Evaluator rollouts (Tracks 2 and 3, recommended) or final reconstructions (Track 1, required). |
| `meshes` | Track 3: the mesh directory passed to the packer. |
| `frame_references` | Track 2 Tier 3: the `--frame-references` directory passed to the packer. |

## Check the container before you send it

Build the image with [examples/container/Dockerfile](examples/container/Dockerfile), then run the verifier inside it:

```bash
docker run --rm --network none -v /path/to/dataset:/data:ro IMAGE \
    /submission/run.sh verify --dataset-root /data
```

It recomputes every hash in the table above from the container and compares it with each submitted file.
It checks that `code/` is a clean checkout of the recorded commit.
It re-packs the stored outputs and compares the result with the submitted file.
Track 2 needs `--dataset-root` for this step.
Every check must pass.

Then reproduce each counted submission and compare it with the submitted file:

```bash
docker run --rm --gpus all --network none -v /path/to/dataset:/data:ro -v "$PWD/out:/output" IMAGE \
    /submission/run.sh reproduce --label tier1 --dataset-root /data --output /output
docker run --rm --network none -v "$PWD/out:/output" IMAGE \
    /submission/run.sh verify --reproduced tier1=/output/tier1/track2_tier1.csv
```

The reproduced file must have the same rows and exactly the same provenance as the submitted one.
The poses will differ slightly.
The FlashCHORD simulator is not bitwise deterministic: float32 atomics in contact resolution differ from run to run, and contact amplifies the differences.
Score the reproduced file locally (the packer prints the scores) and expect it to land within your own run-to-run variation of the leaderboard score.

`--kaggle` also downloads each listed Kaggle submission and compares it with the stored file, when your Kaggle token allows it.
The organizers run this comparison in every case.

## Sending the container

Save the image and record its digest:

```bash
docker save IMAGE | zstd -T0 > TEAM-v2d-submission.tar.zst
sha256sum TEAM-v2d-submission.tar.zst
```

Upload the tarball to Google Drive or another file sharing service.
Submit the link, the tarball's sha256, and the image digest with your technical report.
Make sure the organizers can download the file from the link without requesting access.

## What the organizers check

- The code at the recorded commit is in the container, unmodified, and matches the repository you shared with the organizers.
- Every hash recorded in each counted Kaggle submission matches the container.
- The stored outputs re-pack to the file Kaggle scored.
- Running the entry point reproduces each counted submission, with scores consistent with the leaderboard given the simulator's run-to-run variation.
- The technical report describes the method the container runs.

A container that fails any of these checks can delay or invalidate an award.
