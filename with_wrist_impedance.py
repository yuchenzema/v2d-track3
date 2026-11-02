"""Run a native policy test with an explicitly declared robot impedance profile."""
import json,os,runpy,sys
from pathlib import Path
from wrist_impedance_controller import install
install();target=sys.argv.pop(1);assert target in ['evaluate_reference_controller.py','record_policy_states.py'];runpy.run_path('/episode/scripts/'+target,run_name='__main__')
if target=='record_policy_states.py':
 import numpy as np
 p=os.environ['V2D_STATE_OUTPUT'];data=dict(np.load(p));data['robot_wrist_impedance_profile']=os.environ['V2D_WRIST_IMPEDANCE_PROFILE'];data['controller_protocol']='Learned policy with declared wrist reference impedance and force-preserving positional residual rescaling; continuous unassisted object physics.';np.savez_compressed(p,**data)
else:
 outputs=[x.split('=',1)[1] for x in sys.argv if x.startswith('evaluation.metrics_output=')]
 if outputs:
  p=Path(outputs[-1]);data=json.loads(p.read_text());data['robot_wrist_impedance_profile']=json.loads(os.environ['V2D_WRIST_IMPEDANCE_PROFILE']);p.write_text(json.dumps(data,indent=2))
