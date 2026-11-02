"""Record one complete learned-policy attempt with declared finger correction."""
import os
import runpy
from grip_bias_controller import install
install()
script=('record_hybrid_controller.py' if os.environ.get('V2D_RELEASE_SPEC') else
        'record_velocity_controller.py' if float(os.environ.get('V2D_WRIST_VELOCITY_GAIN','0')) else
        'record_policy_states.py')
runpy.run_path('/episode/scripts/'+script,run_name='__main__')
