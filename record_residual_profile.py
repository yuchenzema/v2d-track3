"""Record a continuous unassisted learned-policy attempt with declared residual gains."""
import json
import os
import runpy
import numpy as np
from residual_gain_controller import install
install()
script = ('record_hybrid_controller.py' if os.environ.get('V2D_RELEASE_SPEC') else
          'record_velocity_controller.py' if float(os.environ.get('V2D_WRIST_VELOCITY_GAIN', '0')) else
          'record_policy_states.py')
runpy.run_path('/episode/scripts/' + script, run_name='__main__')
path = os.environ['V2D_STATE_OUTPUT']
with np.load(path, allow_pickle=False) as archive:
    data = {key: archive[key] for key in archive.files}
assert np.all(data['object_assistance'] == 0)
assert np.array_equal(data['reference_frames'], np.arange(len(data['body_poses'])))
data['residual_gain_profile'] = json.dumps(json.loads(os.environ['V2D_RESIDUAL_PROFILE']), sort_keys=True)
data['controller_protocol'] = 'Learned actor with declared robot-action gains; continuous Newton physics; no object guidance or failure resets.'
np.savez_compressed(path, **data)
