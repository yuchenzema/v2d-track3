import os,runpy,numpy as np
from timed_wrist_impedance_controller import install as impedance
from bimanual_policy_controller import install as actors
impedance();actors();runpy.run_path('/episode/scripts/record_policy_states.py',run_name='__main__')
p=os.environ['V2D_STATE_OUTPUT'];a=dict(np.load(p));a['timed_wrist_impedance_profile']=os.environ['V2D_TIMED_WRIST_IMPEDANCE_PROFILE'];a['actor_combination']=os.environ['V2D_ACTOR_COMBINATION'];a['controller_protocol']='Fixed learned-actor combination with declared source-timed wrist impedance, native45N limits and continuous unassisted object physics.';np.savez_compressed(p,**a)
