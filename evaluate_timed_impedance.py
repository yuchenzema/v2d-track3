import json,os,runpy,sys
from pathlib import Path
from timed_wrist_impedance_controller import install
install();runpy.run_path('/episode/scripts/evaluate_reference_controller.py',run_name='__main__')
for arg in sys.argv:
 if arg.startswith('evaluation.metrics_output='):
  p=Path(arg.split('=',1)[1]);report=json.loads(p.read_text());report['timed_wrist_impedance_profile']=json.loads(os.environ['V2D_TIMED_WRIST_IMPEDANCE_PROFILE']);p.write_text(json.dumps(report,indent=2))
