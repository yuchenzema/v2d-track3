"""Evaluate a declared actor combination with robot command calibration."""
import runpy
from bimanual_policy_controller import install
install()
runpy.run_path('/episode/scripts/evaluate_residual_profile.py',run_name='__main__')
