"""Record genuine learned-policy physics with the declared robot motor controller additions."""
import runpy,os,numpy as np
from wrist_velocity_controller import install as velocity
from release_controller import install as release
velocity();release();runpy.run_path('/episode/scripts/record_policy_states.py',run_name='__main__')
p=os.environ['V2D_STATE_OUTPUT'];data=dict(np.load(p));data['robot_release_controller_spec']=os.environ.get('V2D_RELEASE_SPEC','');data['wrist_reference_velocity_gain']=float(os.environ.get('V2D_WRIST_VELOCITY_GAIN','0'));data['controller_protocol']='Learned residual policy with declared video-timed terminal hand controller and optional wrist velocity feedforward; objects receive no additional forces.';np.savez_compressed(p,**data)
