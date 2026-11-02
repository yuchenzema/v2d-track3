#!/usr/bin/env python3
"""Check a submission container against the files it claims to have produced.

    python3 tools/verify_container.py --manifest /submission/submission.json

Run it inside the submission container. ``submission.json`` lists the code checkout, and for each
leaderboard submission the container stands behind: the exact file uploaded to Kaggle, the policy
checkpoint per episode (Tracks 2 and 3), the evaluator that rolled them out, and the kit copy whose
packer and uploader built and sent the file. ``CONTAINER_SUBMISSION.md`` documents the schema.
This script may come from a newer kit than the one that packed the submissions: packing and
hashing always use the kit named in the manifest.

Every hash recorded in a submitted file must match the artifact in the container:

    eval_script_sha256      sha256 of the evaluate_policy.py that ran
    eval_code_sha256        digest of the FlashCHORD evaluation package (flash_chord/evaluation)
    checkpoint_sha256       sha256 of each episode's checkpoint
    packer_script_sha256    sha256 of tools/pack_submission.py
    uploader_script_sha256  sha256 of submit/v2d_submit.py
    code_commit_url         the commit checked out, unmodified, in the container's code directory

Optional checks:

    --dataset-root DIR     re-pack stored evaluator outputs (Track 2) and compare with the submitted file
    --reproduced L=FILE    compare a freshly reproduced file with submission L (provenance exact,
                           poses summarised; rollouts are not bitwise reproducible)
    --kaggle               download each listed Kaggle submission and compare it with the stored file

Track 1 reconstructions and Track 3 stored outputs are always re-packed and compared.
Exit code 0 when every check passes, 1 otherwise.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import re
import subprocess
import sys
import tempfile
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from v2dlb import report_schema as rs  # noqa: E402

CONFIG = ROOT / "config" / "tracks.json"

#: Mirrors flash_chord.evaluation.harness.evaluation_code_sha256 at the pinned evaluator commit.
EVALUATION_CODE_MODULES = ("__init__.py", "harness.py", "metrics.py", "rollout.py")

POSE_COLUMNS = ["pos_x", "pos_y", "pos_z", "quat_x", "quat_y", "quat_z", "quat_w"]
ROW_EPISODE = re.compile(r"^track_\d+/e0*(\d+)/")
COMMIT_SHA = re.compile(r"/commit/([0-9a-fA-F]{7,40})/?$")


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def evaluation_code_sha256(package: Path) -> str:
    """The evaluator's eval_code_sha256: module names and contents of flash_chord/evaluation."""
    digest = hashlib.sha256()
    for name in sorted(EVALUATION_CODE_MODULES):
        digest.update(name.encode())
        digest.update(b"\0")
        digest.update((package / name).read_bytes())
    return digest.hexdigest()


def read_submission(path: Path) -> pd.DataFrame:
    if path.suffix == ".parquet":
        return pd.read_parquet(path)
    return pd.read_csv(path, dtype={column: str for column in rs.PROVENANCE_COLUMNS})


def same_content(a: pd.DataFrame, b: pd.DataFrame) -> tuple[bool, str]:
    """Equal row sets and values. Numeric columns must agree to float32 rounding."""
    if sorted(a.columns) != sorted(b.columns):
        return False, f"columns differ: {sorted(set(a.columns) ^ set(b.columns))}"
    if len(a) != len(b):
        return False, f"{len(a)} rows vs {len(b)}"
    a = a.sort_values(rs.ROW_ID_COLUMN).reset_index(drop=True)
    b = b.sort_values(rs.ROW_ID_COLUMN).reset_index(drop=True)[a.columns]
    if not a[rs.ROW_ID_COLUMN].equals(b[rs.ROW_ID_COLUMN]):
        return False, "row_id values differ"
    for column in a.columns:
        x, y = a[column], b[column]
        if pd.api.types.is_numeric_dtype(x) and pd.api.types.is_numeric_dtype(y):
            if not np.allclose(x.to_numpy(float), y.to_numpy(float), rtol=1e-6, atol=1e-7, equal_nan=True):
                return False, f"{column}: max difference {np.nanmax(np.abs(x.to_numpy(float) - y.to_numpy(float))):.3g}"
        elif not x.astype(str).str.strip().equals(y.astype(str).str.strip()):
            return False, f"{column}: values differ"
    return True, ""


def episodes_of(frame: pd.DataFrame) -> pd.Series:
    return frame[rs.ROW_ID_COLUMN].astype(str).str.extract(ROW_EPISODE)[0].astype(int)


@dataclass
class Report:
    results: list[dict] = field(default_factory=list)

    def add(self, label: str, check: str, status: str, detail: str = "") -> None:
        self.results.append(dict(submission=label, check=check, status=status, detail=detail))
        print(f"  {status:4s}  {check}{(': ' + detail) if detail else ''}", flush=True)

    @property
    def failed(self) -> bool:
        return any(r["status"] == "FAIL" for r in self.results)


def resolve(base: Path, value: str | None) -> Path | None:
    if not value:
        return None
    path = Path(value)
    return path if path.is_absolute() else base / path


def check_code(report: Report, base: Path, manifest: dict) -> str | None:
    """The code directory is a clean checkout of the manifest's commit. Returns the full sha."""
    code = resolve(base, manifest.get("code", {}).get("path"))
    url = manifest.get("code", {}).get("commit_url", "")
    if not re.match(rs.CODE_COMMIT_PATTERN, url):
        report.add("container", "code.commit_url is a GitHub /commit/<sha> URL", "FAIL", repr(url))
        return None
    if code is None or not (code / ".git").exists():
        report.add("container", "code directory is a git checkout", "FAIL",
                   f"{code} has no .git; ship the repository, not an export")
        return None

    def git(*args: str) -> str:
        # The container may run as a user who does not own the checkout.
        return subprocess.run(["git", "-c", "safe.directory=*", "-C", str(code), *args], check=True,
                              capture_output=True, text=True).stdout.strip()

    try:
        head = git("rev-parse", "HEAD")
        dirty = git("status", "--porcelain", "--untracked-files=no")
    except (OSError, subprocess.CalledProcessError) as error:
        report.add("container", "code directory is a git checkout", "FAIL", str(error).strip()[:200])
        return None
    claimed = COMMIT_SHA.search(url).group(1).lower()
    report.add("container", "checked-out commit matches code.commit_url",
               "PASS" if head.startswith(claimed) else "FAIL", f"HEAD {head[:12]}, manifest {claimed[:12]}")
    report.add("container", "no tracked file differs from the commit",
               "PASS" if not dirty else "FAIL", "" if not dirty else dirty.replace("\n", "; ")[:300])
    return head


def check_provenance(report: Report, label: str, frame: pd.DataFrame, spec: dict, head: str | None) -> None:
    columns, uniform, _ = rs.provenance_for(spec)
    missing = [c for c in columns if c not in frame]
    report.add(label, "provenance columns present", "FAIL" if missing else "PASS", ", ".join(missing))
    for column in uniform:
        if column in frame and frame[column].astype(str).nunique() != 1:
            report.add(label, f"{column} is one value", "FAIL", f"{frame[column].nunique()} values")
    if "code_commit_url" in frame and head:
        url = str(frame["code_commit_url"].iloc[0]).strip()
        match = COMMIT_SHA.search(url)
        ok = bool(match) and head.startswith(match.group(1).lower())
        report.add(label, "submitted code_commit_url is the container's commit", "PASS" if ok else "FAIL",
                   url if not ok else head[:12])


def check_hashes(report: Report, label: str, frame: pd.DataFrame, entry: dict, base: Path) -> None:
    """Tracks 2 and 3: evaluator, packer, uploader and per-episode checkpoint hashes."""
    kit = resolve(base, entry.get("kit")) or ROOT
    evaluator = entry.get("evaluator", {})
    expected = {
        "packer_script_sha256": kit / "tools" / "pack_submission.py",
        "uploader_script_sha256": kit / "submit" / "v2d_submit.py",
        "eval_script_sha256": resolve(base, evaluator.get("script")),
    }
    for column, path in expected.items():
        recorded = str(frame[column].iloc[0]).strip() if column in frame else ""
        if path is None or not path.is_file():
            report.add(label, f"{column} matches the container", "FAIL", f"file not found: {path}")
            continue
        actual = file_sha256(path)
        report.add(label, f"{column} matches the container", "PASS" if actual == recorded else "FAIL",
                   f"{path} {actual[:12]}" + ("" if actual == recorded else f", submitted {recorded[:12]}"))

    package = resolve(base, evaluator.get("evaluation_package"))
    recorded = str(frame["eval_code_sha256"].iloc[0]).strip() if "eval_code_sha256" in frame else ""
    if package is None or not all((package / name).is_file() for name in EVALUATION_CODE_MODULES):
        report.add(label, "eval_code_sha256 matches the container", "FAIL",
                   f"evaluation package not found: {package}")
    else:
        actual = evaluation_code_sha256(package)
        report.add(label, "eval_code_sha256 matches the container", "PASS" if actual == recorded else "FAIL",
                   f"{package} {actual[:12]}" + ("" if actual == recorded else f", submitted {recorded[:12]}"))

    checkpoints = {int(k): resolve(base, v) for k, v in entry.get("checkpoints", {}).items()}
    submitted = frame.assign(_episode=episodes_of(frame)).groupby("_episode")["checkpoint_sha256"].agg(
        lambda s: sorted(set(s.astype(str))))
    problems = []
    for episode, values in submitted.items():
        path = checkpoints.get(int(episode))
        if len(values) != 1:
            problems.append(f"episode {episode}: {len(values)} checkpoint hashes in the file")
        elif path is None or not path.is_file():
            problems.append(f"episode {episode}: no checkpoint in the container")
        elif file_sha256(path) != values[0]:
            problems.append(f"episode {episode}: {path.name} hashes to {file_sha256(path)[:12]}, "
                            f"submitted {values[0][:12]}")
    extra = sorted(set(checkpoints) - set(int(e) for e in submitted.index))
    report.add(label, f"checkpoint_sha256 matches the container for all {len(submitted)} episodes",
               "FAIL" if problems else "PASS", "; ".join(problems[:5]) + (" ..." if len(problems) > 5 else ""))
    if extra:
        report.add(label, "every checkpoint listed is used", "WARN", f"episodes {extra} are not in the file")


def repack(report: Report, label: str, frame: pd.DataFrame, entry: dict, base: Path,
           track: str, commit_url: str, dataset_root: Path | None) -> None:
    """Rebuild the submission from the stored outputs with the container's kit and compare."""
    kit = resolve(base, entry.get("kit")) or ROOT
    outputs = resolve(base, entry.get("outputs"))
    if outputs is None:
        status = "FAIL" if track == "track_1" else "SKIP"
        report.add(label, "re-packed stored outputs match the submitted file", status,
                   "no 'outputs' in the manifest" + ("; Track 1 must ship its final reconstructions"
                                                     if track == "track_1" else ""))
        return
    with tempfile.TemporaryDirectory() as tmp:
        out = Path(tmp) / ("repacked.parquet" if track in ("track_1", "track_3") else "repacked.csv")
        if track == "track_1":
            command = [sys.executable, str(kit / "eval_reconstruction.py"), "--episodes", str(outputs),
                       "--sample", str(kit / "data" / "track_1_sample_submission.parquet"),
                       "--commit", commit_url, "--out", str(out)]
        else:
            if track.startswith("track_2") and dataset_root is None:
                report.add(label, "re-packed stored outputs match the submitted file", "SKIP",
                           "pass --dataset-root to re-pack Track 2 rollouts")
                return
            command = [sys.executable, str(kit / "tools" / "pack_submission.py"), "--track", track,
                       "--results", str(outputs), "--code-commit", commit_url, "-o", str(out),
                       "--skip-crosscheck"]
            if dataset_root is not None:
                command += ["--dataset-root", str(dataset_root)]
            if entry.get("frame_references"):
                command += ["--frame-references", str(resolve(base, entry["frame_references"]))]
            if entry.get("meshes"):
                command += ["--meshes", str(resolve(base, entry["meshes"]))]
            evaluator = resolve(base, entry.get("evaluator", {}).get("script"))
            if evaluator is not None:
                command += ["--eval-script", str(evaluator)]
        result = subprocess.run(command, cwd=kit, capture_output=True, text=True)
        if result.returncode != 0:
            report.add(label, "re-packed stored outputs match the submitted file", "FAIL",
                       (result.stderr or result.stdout).strip().splitlines()[-1][:300])
            return
        ok, detail = same_content(frame, read_submission(out))
        report.add(label, "re-packed stored outputs match the submitted file", "PASS" if ok else "FAIL", detail)


def compare_reproduced(report: Report, label: str, frame: pd.DataFrame, path: Path, spec: dict) -> None:
    """A fresh run: same rows and provenance exactly; poses differ because physics is not bitwise."""
    other = read_submission(path)
    rows_equal = sorted(frame[rs.ROW_ID_COLUMN].astype(str)) == sorted(other[rs.ROW_ID_COLUMN].astype(str))
    report.add(label, "reproduced file has the submitted rows", "PASS" if rows_equal else "FAIL",
               "" if rows_equal else f"{len(other)} rows vs {len(frame)}")
    if not rows_equal:
        return
    a = frame.sort_values(rs.ROW_ID_COLUMN).reset_index(drop=True)
    b = other.sort_values(rs.ROW_ID_COLUMN).reset_index(drop=True)
    columns, _, _ = rs.provenance_for(spec)
    differ = [c for c in columns if c in a and not a[c].astype(str).str.strip().equals(b[c].astype(str).str.strip())]
    report.add(label, "reproduced provenance equals the submitted provenance",
               "FAIL" if differ else "PASS", ", ".join(differ))
    if all(c in a for c in POSE_COLUMNS):
        gap = np.linalg.norm(a[POSE_COLUMNS[:3]].to_numpy(float) - b[POSE_COLUMNS[:3]].to_numpy(float), axis=1)
        report.add(label, "reproduced object positions vs submitted", "INFO",
                   f"median {100 * np.median(gap):.2f} cm, 95th percentile {100 * np.percentile(gap, 95):.2f} cm "
                   "(score the reproduced file to compare with the leaderboard)")


def compare_kaggle(report: Report, label: str, frame: pd.DataFrame, entry: dict, config: dict, track: str) -> None:
    """Each listed Kaggle submission's file has the stored file's content."""
    ids = entry.get("kaggle_submission_ids", {})
    if not ids:
        report.add(label, "Kaggle copies match the stored file", "SKIP", "no kaggle_submission_ids")
        return
    import urllib.request

    from kaggle.api.kaggle_api_extended import KaggleApi
    from kagglesdk.competitions.types.competition_api_service import ApiDownloadSubmissionRequest

    derived = None
    if config["tracks"][track].get("register_object_mesh"):
        from v2dlb.track3_pose import cd_o_submission
        objects = json.loads((ROOT / "data" / "track_3_evaluation_objects.json").read_text())["episodes"]
        derived = cd_o_submission(frame, pd.read_parquet(ROOT / "data" / "track_3_mesh_sample_submission.parquet"),
                                  objects)
    api = KaggleApi()
    api.authenticate()
    with api.build_kaggle_client() as client, tempfile.TemporaryDirectory() as tmp:
        for metric, sid in ids.items():
            request = ApiDownloadSubmissionRequest()
            request.submission_id = int(sid)
            try:
                url = client.competitions.competition_api_client.download_submission(request).url
                suffix = ".parquet" if track in ("track_1", "track_3") else ".csv"
                local = Path(tmp) / f"{sid}{suffix}"
                urllib.request.urlretrieve(url, local)
                copy = read_submission(local)
            except Exception as error:  # noqa: BLE001 - report and carry on with the other boards
                report.add(label, f"Kaggle submission {sid} ({metric}) matches", "WARN",
                           f"could not download ({str(error)[:120]}); the organizers compare it instead")
                continue
            reference = derived if metric == "cd_o_cm" and derived is not None else frame
            ok, detail = same_content(reference, copy)
            report.add(label, f"Kaggle submission {sid} ({metric}) matches", "PASS" if ok else "FAIL", detail)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--manifest", type=Path, default=Path("/submission/submission.json"))
    parser.add_argument("--config", type=Path, default=CONFIG)
    parser.add_argument("--dataset-root", type=Path, help="challenge dataset, to re-pack Track 2 rollouts")
    parser.add_argument("--reproduced", action="append", default=[], metavar="LABEL=FILE",
                        help="a reproduced submission file to compare with submission LABEL")
    parser.add_argument("--kaggle", action="store_true", help="compare with the files Kaggle holds")
    parser.add_argument("--report", type=Path, help="write every check result as JSON")
    args = parser.parse_args()

    config = json.loads(args.config.read_text())
    manifest = json.loads(args.manifest.read_text())
    base = args.manifest.resolve().parent
    reproduced = dict(item.split("=", 1) for item in args.reproduced)
    report = Report()

    print(f"container {manifest.get('team', '?')}")
    head = check_code(report, base, manifest)
    commit_url = manifest.get("code", {}).get("commit_url", "")
    labels = [entry.get("label") for entry in manifest.get("submissions", [])]
    if not labels:
        report.add("container", "at least one submission is listed", "FAIL")
    if len(set(labels)) != len(labels):
        report.add("container", "submission labels are unique", "FAIL", str(labels))
    for unknown in sorted(set(reproduced) - set(labels)):
        report.add(unknown, "--reproduced names a listed submission", "FAIL")

    for entry in manifest.get("submissions", []):
        entry.setdefault("kit", manifest.get("kit"))
        label, track = entry.get("label", "?"), entry.get("track", "")
        print(f"\n{label} ({track})")
        if track not in config["tracks"]:
            report.add(label, "track is a known track", "FAIL", repr(track))
            continue
        spec = config["tracks"][track]
        path = resolve(base, entry.get("file"))
        if path is None or not path.is_file():
            report.add(label, "submitted file is in the container", "FAIL", str(path))
            continue
        frame = read_submission(path)
        report.add(label, "submitted file is in the container", "PASS", f"{path} sha256 {file_sha256(path)[:12]}")
        check_provenance(report, label, frame, spec, head)
        if spec.get("family") == "object_pose":
            check_hashes(report, label, frame, entry, base)
        repack(report, label, frame, entry, base, track, commit_url, args.dataset_root)
        if label in reproduced:
            compare_reproduced(report, label, frame, Path(reproduced[label]), spec)
        if args.kaggle:
            compare_kaggle(report, label, frame, entry, config, track)

    counts = {s: sum(r["status"] == s for r in report.results) for s in ("PASS", "FAIL", "WARN", "SKIP")}
    print("\n" + ", ".join(f"{n} {s.lower()}" for s, n in counts.items()))
    if args.report:
        args.report.write_text(json.dumps(dict(manifest=str(args.manifest), results=report.results), indent=1) + "\n")
    return 1 if report.failed else 0


if __name__ == "__main__":
    sys.exit(main())
