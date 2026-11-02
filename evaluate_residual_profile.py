"""Run the native evaluator with declared gains on learned robot actions."""
import runpy
from residual_gain_controller import install
install()
runpy.run_path('/episode/scripts/evaluate_reference_controller.py', run_name='__main__')
