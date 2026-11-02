import runpy
from bimanual_policy_controller import install
install()
runpy.run_path('/episode/scripts/evaluate_reference_controller.py',run_name='__main__')
