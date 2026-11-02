"""Stage genuine recorded policy poses on the public Track 3 frame roster.

This is a local export adapter, not another policy evaluation or a challenge score.
It never changes training inputs, chooses a different attempt, repairs a failed
pose, or uploads anything. Full physics recordings and their hashes remain the
source of truth. Run with .venv-submission/bin/python.
"""
from __future__ import annotations

import argparse
import fcntl
import hashlib
import json
import os
from pathlib import Path
import re
import sys
import time

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq
import trimesh

BASE = Path(__file__).resolve().parent
BATCH = BASE / "runs/evaluation_batch_20261015"
OUT = BATCH / "submission_staging"
KIT = BASE / "v2d_submission_kit"
sys.path.insert(0, str(KIT))
from v2dlb.mesh_budget import budget_mesh
from v2dlb.submission_parquet import read_trajectory
from batch_artifacts import complete


def digest(path):
    h = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1 << 20), b""):
            h.update(block)
    return h.hexdigest()


def write_json(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + f".{os.getpid()}.tmp")
    temporary.write_text(json.dumps(value, indent=2))
    temporary.replace(path)


def source(ep):
    if ep == 1:
        # The user explicitly preferred the earlier policy. Do not substitute
        # the subsequent v3/v4/v5 actor because its run number is higher.
        root = BASE / "runs/episode_000001_newton_v2_20261014"
        recording_root = BASE / "runs/episode_000001_newton_v5_20261014"
        return {
            "root": root,
            "recording_root": recording_root,
            "state": recording_root / "evaluation/baseline-policy-states.npz",
            "checkpoint": root / "flash_chord/policy_250003456.safetensors",
            "reason": "User preferred the earlier episode 1 policy; retained v2 actor, recorded as the baseline in v5.",
            "quality": {"exact_human_motion_match_verified": False},
            "selection": None,
        }
    recommendation_path = BATCH / "improvements_20261019/best-policy-sources.json"
    if ep != 5 and recommendation_path.exists():
        recommendation = json.loads(recommendation_path.read_text())
        row = recommendation.get("episodes", {}).get(str(ep)) if recommendation.get("finalized") else None
        if row and row["method"] != "preserved_baseline":
            root = Path(row["run"])
            selected = row["selected"]
            return {
                "root": root, "recording_root": root,
                "state": root / selected["recorded_states"],
                "checkpoint": root / selected["checkpoint"].removeprefix("/episode/"),
                "reason": "Final verified comparison selected a measured improvement; original baseline remains available.",
                "quality": row["quality_assessment"], "selection": None,
                "variant_method": row["method"],
                "release_kind": row.get("release_kind"),
                "expected_state_sha256": row["recorded_states_sha256"],
                "expected_checkpoint_sha256": row["checkpoint_sha256"],
                "residual_gain_profile": selected.get("residual_gain_profile"),
                "reproduction_controls": row.get("reproduction_controls"),
                "actor_combination": selected.get("actor_combination"),
            }
    root = (BASE / "runs/episode_000005_newton_20261015" if ep == 5
            else BASE / "runs" / f"episode_{ep:06d}_newton_batch_20261015")
    if ep != 5 and not complete(root):
        return None
    selection_path = root / "final-controller-selection.json"
    if not selection_path.exists() or not (root / "result-report.json").exists():
        return None
    selection = json.loads(selection_path.read_text())["selected"]
    report = json.loads((root / "result-report.json").read_text())
    return {
        "root": root,
        "recording_root": root,
        "state": root / selection["recorded_states"],
        "checkpoint": root / selection["checkpoint"].removeprefix("/episode/"),
        "reason": "Whole physical attempt selected by the existing checkpoint/controller review.",
        "quality": report.get("quality_assessment", {"exact_human_motion_match_verified": False}),
        "selection": selection_path,
    }


def public_grid():
    names = json.loads((KIT / "data/track_3_evaluation_objects.json").read_text())["episodes"]
    groups = {}
    for row_id in pq.read_table(KIT / "data/track_3_sample_submission.parquet",
                                columns=["row_id"])["row_id"].to_pylist():
        match = re.fullmatch(r"track_3/e(\d{6})/w(\d{3})/f(\d{6})/o(\d+)", row_id)
        if match is None:
            raise ValueError(f"Invalid public row ID: {row_id}")
        ep, world, frame, slot = map(int, match.groups())
        groups.setdefault(ep, []).append((world, frame, slot))
    if set(groups) != set(map(int, names)):
        raise ValueError("Public sample and object roster disagree")
    return groups, names


def code_bundle(sources):
    files = {Path(__file__), BASE / "batch_artifacts.py", BASE / "prepare_submission_spoon_mesh.py"}
    for item in sources.values():
        if item is None:
            continue
        scripts = item["recording_root"] / "scripts"
        files.update(scripts.glob("*.py"))
    content = {str(p.relative_to(BASE)): digest(p) for p in sorted(files)}
    value = hashlib.sha256(json.dumps(content, sort_keys=True).encode()).hexdigest()
    write_json(OUT / "export-code-bundle.json", {"sha256": value, "source_files": content})
    return value


def stage_episode(ep, item, rows, object_names, code_hash, previous):
    if sorted(set(world for world, _, _ in rows)) != [0]:
        raise ValueError("This adapter exports one continuous physical world; the roster requests another grid")
    frames = sorted(set(frame for _, frame, _ in rows))
    if frames != list(range(frames[-1] + 1)):
        raise ValueError("Unexpected non-contiguous public source frame grid")
    T = len(frames)
    identity = {
        "source_state_sha256": digest(item["state"]),
        "checkpoint_sha256": digest(item["checkpoint"]),
        "eval_script_sha256": digest(Path(__file__)),
        "eval_code_sha256": code_hash,
        "source_frames": T,
        "object_names": object_names,
        "mesh_sha256": {name: digest(item["root"] / f"reconstruction/mesh_priors/{name}.ply")
                        for name in object_names},
        "selection_sha256": digest(item["selection"]) if item["selection"] else None,
        "variant_method": item.get("variant_method", "preserved_baseline"),
        "declared_residual_gain_profile": item.get("residual_gain_profile"),
        "reproduction_controls": item.get("reproduction_controls"),
        "actor_combination": item.get("actor_combination"),
    }
    if item.get("expected_state_sha256") and identity["source_state_sha256"] != item["expected_state_sha256"]:
        raise ValueError("Selected improvement recording no longer matches its final comparison hash")
    if item.get("expected_checkpoint_sha256") and identity["checkpoint_sha256"] != item["expected_checkpoint_sha256"]:
        raise ValueError("Selected improvement checkpoint no longer matches its final comparison hash")
    combination = item.get("actor_combination")
    if combination:
        alternate = item["root"] / combination["alternate_checkpoint"].removeprefix("/episode/")
        if digest(alternate) != combination["alternate_checkpoint_sha256"]:
            raise ValueError("Selected alternate actor no longer matches its declared checksum")
    if item["selection"]:
        selected = json.loads(item["selection"].read_text())["selected"]
        expected = selected.get("checkpoint_sha256")
        if expected and identity["checkpoint_sha256"] != expected:
            raise ValueError("Selected checkpoint no longer matches its recorded review hash")
    manifest_path = item["root"] / "deliverables/manifest.json"
    if ep not in (1, 5) and item.get("release_kind") not in ("checkpoint", "controller") and item.get("variant_method") not in ("declared_residual_gains", "fixed_hand_actor_combination", "declared_grip_correction") and manifest_path.exists():
        manifest = json.loads(manifest_path.read_text())
        if identity["source_state_sha256"] != manifest["policy_states.npz"]["sha256"]:
            raise ValueError("Physical recording no longer matches the validated deliverables manifest")
    result_path = OUT / "results" / f"episode_{ep:06d}.parquet"
    if (previous.get("identity") == identity and result_path.exists()
            and previous.get("trajectory_sha256") == digest(result_path)
            and all((OUT / "meshes" / f"episode_{ep:06d}" / f"{n}.ply").exists()
                    and previous.get("meshes", {}).get(n, {}).get("submitted_sha256")
                    == digest(OUT / "meshes" / f"episode_{ep:06d}" / f"{n}.ply")
                    for n in object_names)):
        return previous
    with np.load(item["state"], allow_pickle=False) as state:
        poses = state["body_poses"]
        N = len(poses)
        if N != 2 * (T - 1):
            raise ValueError(f"Expected half-speed source clock {2 * (T - 1)} controls, received {N}")
        if not np.isfinite(poses).all():
            raise ValueError("Recorded attempt has non-finite states")
        if not np.all(state["object_assistance"] == 0):
            raise ValueError("Recorded attempt used object assistance")
        if not np.array_equal(state["reference_frames"], np.arange(N)):
            raise ValueError("Recorded clock skipped/repeated frames; refusing to export")
        labels = state["body_names"].tolist()
        body_ids = [labels.index(f"track3_{name}/object") for name in object_names]
        sample = np.rint(np.arange(T) * (N - 1) / (T - 1)).astype(np.int64)
        selected = np.array([poses[sample[frame], body_ids[slot]]
                             for _, frame, slot in rows])
    if not np.allclose(np.linalg.norm(selected[:, 3:], axis=1), 1, atol=1e-4):
        raise ValueError("Recorded object quaternions are not unit XYZW rotations")
    columns = {
        "world_index": np.array([w for w, _, _ in rows], dtype=np.int32),
        "frame_index": np.array([f for _, f, _ in rows], dtype=np.int32),
        "object_slot": np.array([s for _, _, s in rows], dtype=np.int32),
        "object_name": [object_names[s] for _, _, s in rows],
    }
    for axis, name in enumerate(["pos_x", "pos_y", "pos_z", "quat_x", "quat_y", "quat_z", "quat_w"]):
        columns[name] = selected[:, axis]
    metadata = {
        "episode_index": str(ep),
        "quaternion_layout": "xyzw", "position_unit": "meter",
        "eval_script_sha256": identity["eval_script_sha256"],
        "eval_code_sha256": code_hash,
        "checkpoint_sha256": identity["checkpoint_sha256"],
        "v2d.export_method": "Genuine single-attempt Newton body poses, uniformly sampled at source-video frame times",
        "v2d.source_state_sha256": identity["source_state_sha256"],
        "v2d.control_frame_count": str(N),
        "v2d.source_frame_count": str(T),
        "v2d.motion_speed": "0.5",
        "v2d.source_control_mapping": "round(i*(N-1)/(T-1)); same mapping as published comparison videos",
        "v2d.provenance_note": "eval_script/code hashes identify this export adapter and its recording-code bundle; no claim of a new official evaluator run",
        "v2d.policy_variant": identity["variant_method"],
        "v2d.declared_residual_gain_profile": json.dumps(item.get("residual_gain_profile"), sort_keys=True),
        "v2d.reproduction_controls": json.dumps(item.get("reproduction_controls"), sort_keys=True),
        "v2d.actor_combination": json.dumps(item.get("actor_combination"), sort_keys=True),
    }
    table = pa.table(columns).replace_schema_metadata({k.encode(): v.encode() for k, v in metadata.items()})
    result_path.parent.mkdir(parents=True, exist_ok=True)
    temporary = result_path.with_suffix(".tmp")
    pq.write_table(table, temporary)
    # Use the actual kit reader, and compare its decoded values to the real
    # physics samples rather than trusting only our own output column names.
    decoded = read_trajectory(temporary)
    actual_keys = list(zip(decoded.world_index.tolist(), decoded.frame_index.tolist(), decoded.object_slot.tolist()))
    if decoded.episode_index != ep or actual_keys != rows or not np.array_equal(decoded.pose_xyzw, selected):
        raise ValueError("Kit trajectory reader does not reproduce the exact requested poses/grid")
    temporary.replace(result_path)
    mesh_details = {}
    for name in object_names:
        path = item["root"] / f"reconstruction/mesh_priors/{name}.ply"
        mesh = trimesh.load(path, force="mesh", process=False)
        if not np.isfinite(mesh.vertices).all() or len(mesh.faces) == 0 or mesh.area <= 0:
            raise ValueError(f"Invalid reconstructed surface: {path}")
        # Exercise the same 4096/4096 decimation and validation path as the
        # official packer. The original body-frame mesh is left untouched.
        submitted_path = path
        preparation = None
        try:
            vertices, faces = budget_mesh(path, faces=4096, vertices=4096)
        except ValueError as error:
            if name != "wooden_spoon":
                raise
            from prepare_submission_spoon_mesh import prepare
            submitted_path = OUT / "prepared_meshes" / f"episode_{ep:06d}" / f"{name}.ply"
            preparation = prepare(path, submitted_path)
            preparation["original_packer_failure"] = str(error)
            vertices, faces = budget_mesh(submitted_path, faces=4096, vertices=4096)
        if (vertices.shape != (4096, 3) or faces.shape != (4096, 3)
                or not np.isfinite(vertices).all() or faces.min() < 0 or faces.max() >= len(vertices)):
            raise ValueError(f"Official mesh budget validation failed: {name}")
        target = OUT / "meshes" / f"episode_{ep:06d}" / f"{name}.ply"
        target.parent.mkdir(parents=True, exist_ok=True)
        if target.is_symlink():
            target.unlink()
        elif target.exists():
            raise ValueError(f"Refusing to replace a non-symlink mesh: {target}")
        target.symlink_to(submitted_path)
        mesh_details[name] = {"body_label": f"track3_{name}/object", "source": str(path),
                              "sha256": identity["mesh_sha256"][name], "units": "meters",
                              "submitted_mesh": str(submitted_path),
                              "submitted_sha256": digest(submitted_path),
                              "submission_mesh_preparation": preparation,
                              "body_frame_transform_added": False,
                              "official_4096_vertex_face_budget_validated": True}
    result = {
        "status": "staged_locally", "episode": ep, "identity": identity,
        "trajectory": str(result_path), "trajectory_sha256": digest(result_path),
        "recording": str(item["state"]), "checkpoint": str(item["checkpoint"]),
        "recorded_control_frames": N, "source_frames": T, "rows": len(rows),
        "source_control_indices": sample.tolist(), "meshes": mesh_details,
        "selection_reason": item["reason"], "quality": item["quality"],
        "no_pose_edits": True, "no_dtw_or_failure_splicing": True,
        "same_sampling_as_published_source_comparison": True,
        "kit_trajectory_reader_verified": True,
        "official_challenge_score_available": False,
    }
    write_json(OUT / "provenance" / f"episode_{ep:06d}.json", result)
    return result


def update():
    grid, names = public_grid()
    sources = {ep: source(ep) for ep in sorted(grid)}
    code_hash = code_bundle(sources)
    report_path = OUT / "readiness.json"
    old = json.loads(report_path.read_text()) if report_path.exists() else {}
    episodes = {}
    for ep, item in sources.items():
        key = str(ep)
        if item is None:
            episodes[key] = {"status": "waiting_for_training_selection_and_export"}
            continue
        try:
            episodes[key] = stage_episode(ep, item, grid[ep], names[key], code_hash,
                                          old.get("episodes", {}).get(key, {}))
        except Exception as error:
            episodes[key] = {"status": "needs_attention", "error": f"{type(error).__name__}: {error}"}
    staged = sum(row["status"] == "staged_locally" for row in episodes.values())
    report = {
        "updated_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "scored_episodes_required": len(grid), "episodes_staged": staged,
        "all_20_episode_data_staged": staged == len(grid),
        "final_submission_package_ready": False, "uploaded": False,
        "remaining_submission_requirements": [
            ("Review the remaining motion discrepancies in the per-episode reports."
             if staged == len(grid) else "Finish and validate every scored episode before building the full submission."),
            "Pin the producing implementation to its actual GitHub code commit; no placeholder or upstream commit is substituted.",
            "Run the official full-roster packer with that code commit and validate the combined trajectory/mesh Parquet.",
        ],
        "quality_note": "Local artifact readiness is independent of successful imitation; per-episode reports retain tracking failures. No confidential ground truth or official leaderboard score is available locally.",
        "export_provenance_note": "This stages legacy-format trajectories accepted by the local kit reader. Adapter hashes and original physical-recording/code hashes are retained separately; this is not a new organizer-evaluator run.",
        "episodes": episodes,
    }
    write_json(report_path, report)
    print(json.dumps({k: report[k] for k in ["updated_utc", "scored_episodes_required", "episodes_staged", "final_submission_package_ready"]}), flush=True)
    return staged


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--watch", action="store_true")
    args = parser.parse_args()
    OUT.mkdir(parents=True, exist_ok=True)
    with (OUT / "stage.lock").open("a") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        while True:
            try:
                staged = update()
            except (OSError, ValueError, KeyError) as error:
                if not args.watch:
                    raise
                print(f"Staging will retry after concurrent artifact update: {error}", flush=True)
                staged = 0
            phase_path = BATCH / "improvements_20261019/phase-status.json"
            phase_finished = (not phase_path.exists() or json.loads(phase_path.read_text()).get("phase")
                              in ("completed", "completed_with_experiment_failures"))
            if not args.watch or staged == 20 and phase_finished:
                break
            time.sleep(30)
