"""Run a standard policy probe with explicitly declared physical wrist motor gains."""
import runpy,sys
from pathlib import Path
from wrist_motor_controller import install
install()
target=sys.argv.pop(1)
assert target in ['evaluate_reference_controller.py','evaluate_residual_profile.py','record_policy_states.py','record_residual_profile.py','record_velocity_controller.py']
runpy.run_path(str(Path('/episode/scripts')/target),run_name='__main__')
