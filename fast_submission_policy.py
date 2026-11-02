"""Explicit user authorization for completing a valid cohort before refinement."""
import json
from pathlib import Path
from improvement_common import BATCH


def policy(out=None):
    path=(Path(out) if out is not None else BATCH/'focused_episodes_20261028')/'own-geometry-policy.json'
    try:return json.loads(path.read_text())
    except (OSError,ValueError):return {}


def enabled(out=None):
    data=policy(out)
    return data.get('enabled') is True and data.get('submission_mode')=='complete_first'
