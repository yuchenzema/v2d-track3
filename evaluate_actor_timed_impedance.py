import json,os,runpy,sys
from pathlib import Path
from timed_wrist_impedance_controller import install as impedance
from bimanual_policy_controller import install as actors
impedance();actors();runpy.run_path('/episode/scripts/evaluate_reference_controller.py',run_name='__main__')
for arg in sys.argv:
 if arg.startswith('evaluation.metrics_output='):
  p=Path(arg.split('=',1)[1]);a=json.loads(p.read_text());a['timed_wrist_impedance_profile']=json.loads(os.environ['V2D_TIMED_WRIST_IMPEDANCE_PROFILE']);a['actor_combination']=json.loads(os.environ['V2D_ACTOR_COMBINATION']);p.write_text(json.dumps(a,indent=2))
