"""Evaluate one declared finger correction across the full native cohort."""
import runpy
from grip_bias_controller import install
install()
runpy.run_path('/episode/scripts/evaluate_reference_controller.py',run_name='__main__')
