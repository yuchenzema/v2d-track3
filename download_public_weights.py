#!/usr/bin/env python3
"""Download public Track 3 baseline weights without requiring Docker.

Gated models and models requiring explicit license acceptance are excluded.
Run with v2d-track3/.venv/bin/python; completed downloads are reusable.
"""
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
import hashlib
import json
import os
import runpy
import subprocess
import sys

from huggingface_hub import snapshot_download

ROOT = Path(__file__).resolve().parent
MODULES = ROOT / "video_to_data/reconstruction/modules"
WEIGHTS = ROOT / "video_to_data/reconstruction/data/weights"
REPORT = ROOT / "downloads/public-weights-status.json"


def hf(repo, directory, patterns=None, cache_dir=None, repo_type="model"):
    return snapshot_download(repo, repo_type=repo_type, local_dir=str(directory) if directory else None,
                             allow_patterns=patterns, cache_dir=cache_dir,
                             max_workers=2)


def url_download(url, destination, sha256=None):
    destination = Path(destination)
    destination.parent.mkdir(parents=True, exist_ok=True)
    if not destination.is_file():
        temporary = destination.with_suffix(destination.suffix + ".partial")
        subprocess.run(["curl", "-fL", "--retry", "3", "--connect-timeout", "30",
                        "--max-time", "1800", "-C", "-", url, "-o", str(temporary)],
                       check=True)
        temporary.replace(destination)
    if sha256:
        digest = hashlib.file_digest(destination.open("rb"), "sha256").hexdigest() if sys.version_info >= (3, 11) else None
        if digest is None:
            h = hashlib.sha256()
            with destination.open("rb") as stream:
                for block in iter(lambda: stream.read(1024 * 1024), b""):
                    h.update(block)
            digest = h.hexdigest()
        if digest != sha256:
            raise RuntimeError(f"Checksum mismatch: {destination}")
    return str(destination)


def upstream(module, function, **kwargs):
    path = MODULES / f"v2d_{module}/lib/download_weights.py"
    return runpy.run_path(str(path))[function](**kwargs)


tasks = {
    "GR00T-N1.7-3B": lambda: hf("nvidia/GR00T-N1.7-3B", ROOT / "models/GR00T-N1.7-3B"),
    "MoGE-2": lambda: hf("Ruicheng/moge-2-vitl-normal", WEIGHTS / "moge"),
    "SAM2.1": lambda: hf("facebook/sam2.1-hiera-large", WEIGHTS / "sam2"),
    "GroundingDINO": lambda: upstream("grounding_dino", "download_weights", output_dir=str(WEIGHTS / "grounding_dino")),
    "HaMeR": lambda: hf("geopavlakos/HaMeR", WEIGHTS / "hamer", ["_DATA/hamer_ckpts/**", "_DATA/vitpose_ckpts/**", "_DATA/data/mano_mean_params.npz"], repo_type="space"),
    "DROID-SLAM": lambda: upstream("droid_slam", "download_droid", output_dir=str(WEIGHTS / "droid_slam")),
    "WiLoR": lambda: hf("warmshao/WiLoR-mini", WEIGHTS / "wilor", ["pretrained_models/wilor_final.ckpt", "pretrained_models/detector.pt", "pretrained_models/mano_mean_params.npz"]),
    "AnyCalib": lambda: url_download("https://github.com/javrtg/AnyCalib/releases/download/v1.0.0/anycalib_gen.pt", WEIGHTS / "anycalib/hub/anycalib/anycalib_gen.pt"),
    "GeoCalib": lambda: url_download("https://github.com/cvg/GeoCalib/releases/download/v1.0/geocalib-pinhole.tar", WEIGHTS / "geocalib/hub/geocalib/pinhole.tar"),
    "VGG16-gsplat": lambda: url_download("https://download.pytorch.org/models/vgg16-397923af.pth", WEIGHTS / "gsplat_refinement/vgg16-397923af.pth", "397923af8e79cdbb6a7127f12361acd7a2f83e06b05044ddf496e83de57a5bf0"),
    "SAM3D-MoGE1": lambda: hf("Ruicheng/moge-vitl", None, cache_dir=str(WEIGHTS / "sam3d/hf_home/hub")),
    "SAM3D-DINOv2-L": lambda: url_download("https://dl.fbaipublicfiles.com/dinov2/dinov2_vitl14/dinov2_vitl14_reg4_pretrain.pth", WEIGHTS / "sam3d/torch_home/hub/checkpoints/dinov2_vitl14_reg4_pretrain.pth"),
    "SAM3D-DINOv2-B": lambda: url_download("https://dl.fbaipublicfiles.com/dinov2/dinov2_vitb14/dinov2_vitb14_reg4_pretrain.pth", WEIGHTS / "sam3d/torch_home/hub/checkpoints/dinov2_vitb14_reg4_pretrain.pth"),
}

if __name__ == "__main__":
    os.environ["PATH"] = str(ROOT / ".venv/bin") + ":" + os.environ["PATH"]
    results = {}
    with ThreadPoolExecutor(max_workers=2) as pool:
        futures = {pool.submit(task): name for name, task in tasks.items()}
        for future in as_completed(futures):
            name = futures[future]
            try:
                future.result()
                results[name] = {"status": "downloaded"}
            except Exception as exc:
                results[name] = {"status": "failed", "error": str(exc)}
            REPORT.write_text(json.dumps(results, indent=2) + "\n")
            print(name, results[name]["status"], flush=True)
    raise SystemExit(any(item["status"] == "failed" for item in results.values()))
