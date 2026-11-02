"""Record the same actor combination and robot command calibration as evaluation."""
import json,os,runpy,numpy as np
from bimanual_policy_controller import install
install()
runpy.run_path('/episode/scripts/record_residual_profile.py',run_name='__main__')
p=os.environ['V2D_STATE_OUTPUT'];a=dict(np.load(p,allow_pickle=False));a['actor_combination']=json.dumps(json.loads(os.environ['V2D_ACTOR_COMBINATION']),sort_keys=True)
a['controller_protocol']='Declared learned-actor combination and source-phase robot motor calibration; continuous unassisted Newton physics.'
np.savez_compressed(p,**a)
