"""Score a named reconstruction candidate on all original source frames."""
import argparse, subprocess
from pathlib import Path
p=argparse.ArgumentParser();p.add_argument('--run',type=Path,required=True);p.add_argument('--candidate',required=True);p.add_argument('--label',required=True);a=p.parse_args();r=a.run.resolve()
assert Path(a.candidate).name==a.candidate and Path(a.label).name==a.label
source=(r/'scripts/assess_source_fidelity.py').read_text()
old="REC/'scene_motion_physical.npz'";assert old in source
source=source.replace(old,f"REC/{a.candidate!r}")
script=r/f'scripts/assess_{a.label.replace("-","_")}.py';script.write_text(source)
subprocess.run([str(r/'.venv-video/bin/python'),str(script),'--output',a.label,'--overlay'],check=True)
