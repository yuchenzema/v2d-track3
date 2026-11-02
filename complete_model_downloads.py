#!/usr/bin/env python3
"""Resume account-gated downloads and optionally place a licensed MANO archive."""
import argparse
import json
from pathlib import Path
import zipfile

from huggingface_hub import get_token, snapshot_download

ROOT = Path(__file__).resolve().parent
WEIGHTS = ROOT / "video_to_data/reconstruction/data/weights"


def install_mano(archive):
    with zipfile.ZipFile(archive) as source:
        for name in ("MANO_LEFT.pkl", "MANO_RIGHT.pkl"):
            matches = [item for item in source.namelist()
                       if Path(item).name == name and "__MACOSX" not in item]
            if len(matches) != 1:
                raise ValueError(f"Expected exactly one {name} in the licensed archive")
            content = source.read(matches[0])
            targets = [ROOT / "models/mano/models", WEIGHTS / "hand/models",
                       WEIGHTS / "hamer/_DATA/data/mano"]
            if name == "MANO_RIGHT.pkl":
                targets.append(WEIGHTS / "wilor/pretrained_models")
            for target in targets:
                target.mkdir(parents=True, exist_ok=True)
                destination = target / name
                if destination.exists() and destination.read_bytes() != content:
                    raise ValueError(f"Existing MANO model differs: {destination}")
                destination.write_bytes(content)
    print("MANO models placed for the task loader, hand alignment, HaMeR and WiLoR.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--mano-zip", type=Path,
                        help="MANO archive obtained from https://mano.is.tue.mpg.de/")
    args = parser.parse_args()
    if args.mano_zip:
        install_mano(args.mano_zip)
    report = {}
    if not get_token():
        print("Log in locally: ~/v2d-track3/.venv/bin/hf auth login")
        print("Model access is also required at:")
        print("https://huggingface.co/facebook/sam-3d-objects")
        print("https://huggingface.co/nvidia/Cosmos-Reason2-2B")
        raise SystemExit(1)
    for name, directory in [
        ("facebook/sam-3d-objects", WEIGHTS / "sam3d/hf-download"),
        ("nvidia/Cosmos-Reason2-2B", ROOT / "models/Cosmos-Reason2-2B"),
    ]:
        try:
            if name.endswith("Cosmos-Reason2-2B"):
                # GR00T uses this model ID internally; prime its normal HF cache
                # without storing a second copy of the large safetensors file.
                snapshot = snapshot_download(name, max_workers=2)
                if not directory.exists():
                    directory.symlink_to(snapshot, target_is_directory=True)
            else:
                snapshot_download(name, local_dir=str(directory), max_workers=2)
            report[name] = "downloaded"
        except Exception as exc:
            report[name] = type(exc).__name__
            print(name, "could not be downloaded; check model access.")
        (ROOT / "downloads/gated-model-status.json").write_text(json.dumps(report, indent=2) + "\n")
    raise SystemExit(any(status != "downloaded" for status in report.values()))
