import os,runpy,numpy as np
from timed_wrist_impedance_controller import install
install();runpy.run_path('/episode/scripts/record_policy_states.py',run_name='__main__')
p=os.environ['V2D_STATE_OUTPUT'];data=dict(np.load(p));data['timed_wrist_impedance_profile']=os.environ['V2D_TIMED_WRIST_IMPEDANCE_PROFILE'];data['controller_protocol']='Learned actor with declared source-timed robot wrist impedance at physics substeps; native45N limits, zero object assistance.';np.savez_compressed(p,**data)
