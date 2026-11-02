"""Evaluate a declared alternate reference with native schema checks and zero object assistance."""
import os,runpy
from pathlib import Path
import json
if json.loads(Path('/episode/episode-config.json').read_text()).get('validated_video_pipeline'):
 from legacy_source_frame_playback import install as install_source_frame_playback
 install_source_frame_playback()
if os.environ.get('V2D_ROBOT_WRIST_CALIBRATION'):
 from flash_chord.envs.base import BaseEnv
 if not getattr(BaseEnv,'_wrist_calibration_installed',False):
  from robot_wrist_calibration import install as install_calibration
  install_calibration()
if os.environ.get('V2D_BRISTLE_COMPLIANCE'):
 from newton.solvers import SolverMuJoCo
 if not getattr(SolverMuJoCo,'_brush_tip_material_installed',False):
  from brush_bristle_compliance import install as install_compliance
  install_compliance()
from wrist_velocity_controller import install
install()
if os.environ.get('V2D_ROBOT_TRACKING_FEEDBACK') or os.environ.get('V2D_ROBOT_TRACKING_PROFILES'):
 from flash_chord.envs.base import BaseEnv
 if not getattr(BaseEnv,'_robot_tracking_feedback_installed',False):
  from robot_object_tracking_feedback import install as install_tracking
  install_tracking()
if os.environ.get('V2D_RELEASE_SPEC'):
 from release_controller import install as install_release
 install_release()
reference=os.environ.get('V2D_REFERENCE_OVERRIDE')
if reference:
 import flash_chord.evaluation.harness as harness
 original=harness._resolve_parquet
 def resolve(config,source_root):
  path=Path(reference);assert path.is_dir() and (path/'motion.parquet').exists();return path
 harness._resolve_parquet=resolve
 print('Declared inference reference override:',reference,flush=True)
substeps=os.environ.get('V2D_SIM_SUBSTEPS')
capacities={key:int(os.environ[name]) for key,name in [('nconmax','V2D_SIM_NCONMAX'),('njmax','V2D_SIM_NJMAX')] if os.environ.get(name)}
decompose=os.environ.get('V2D_SCENE_DECOMPOSE_OBJECTS')
if decompose is not None:assert decompose in ['true','false']
if substeps or capacities or decompose is not None:
 import flash_chord.evaluation.harness as harness
 native_config=harness.checkpoint_config
 def configuration(metadata):
  config=native_config(metadata)
  values=dict(capacities)
  if substeps:values['substeps']=int(substeps)
  for section in [config['sim'],config['env']['sim']]:section.update(values)
  if decompose is not None:config['scene']['decompose_objects']=decompose=='true'
  return config
 harness.checkpoint_config=configuration
 print('Declared inference physics parameters:',dict(substeps=substeps,**capacities),flush=True)
runpy.run_path('/workspace/scripts/evaluate_policy.py',run_name='__main__')
