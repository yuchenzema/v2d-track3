"""Declared source-transfer diagnostic with native checkpoint schema validation.

Retain the checkpoint's actor and motor/action semantics. Substitute only the
new episode's reference and unit source clock. Outputs are research diagnostics,
never official submission provenance or Kaggle scores.
"""
import copy
import json
import os
from pathlib import Path
import runpy
import sys

sys.path.insert(0,'/episode/scripts')
from legacy_source_frame_playback import install
install()
import flash_chord.evaluation.harness as harness

root=Path('/episode');episode=json.loads((root/'episode-config.json').read_text())
original=harness.checkpoint_config


def source_configuration(metadata):
    config=copy.deepcopy(original(metadata))
    config['task'].update(parquet=f"/episode/human_motion_data/{episode['training_reference']}/processed/sequence_id={episode['sequence']}/robot_name=sharpa_wave",
                          motion_speed=1.,source_frame_playback=True)
    if os.environ.get('V2D_TRANSFER_NATIVE_PHYSICS')=='true':
        for section in [config['sim'],config['env']['sim']]:
            section.update(substeps=episode['physics_substeps'],nconmax=2048,njmax=4096)
    return config


harness.checkpoint_config=source_configuration
runpy.run_path('/workspace/scripts/evaluate_policy.py',run_name='__main__')
