"""Full source-clock PD feasibility test; never label it a learned policy."""
import json
import os
from pathlib import Path
import runpy
from wrist_motor_controller import install
install()
if os.environ.get('V2D_WRIST_VELOCITY_GAIN'):
 from wrist_velocity_controller import install as install_velocity
 install_velocity()
try:
    runpy.run_path('/episode/scripts/diagnose_zero_residual.py',run_name='__main__')
except SystemExit as outcome:
    if outcome.code not in [None,0]:raise
root=Path('/episode');label=os.environ['V2D_DIAGNOSTIC_LABEL']
path=root/f'evaluation/{label}-protocol.json';report=json.loads(path.read_text())
report.update(wrist_motor_profile=json.loads(os.environ['V2D_WRIST_MOTOR_PROFILE']),
    wrist_reference_velocity_gain=float(os.environ.get('V2D_WRIST_VELOCITY_GAIN','0.')),
    wrist_side_motor_profiles=json.loads(os.environ.get('V2D_WRIST_SIDE_MOTOR_PROFILES','{}')),
    controller_actuator_force_limits_unchanged=True,policy_inference=False,
    interpretation='Reference feasibility only; training and a genuine learned-policy rollout remain required')
path.write_text(json.dumps(report,indent=2))
