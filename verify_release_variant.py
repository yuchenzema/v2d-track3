"""Verify pose invariants and release continuity after the native collision pass."""
from pathlib import Path
import argparse,json
import numpy as np,pyarrow.parquet as pq
from scipy.spatial.transform import Rotation
p=argparse.ArgumentParser();p.add_argument('--run',type=Path,required=True);p.add_argument('--baseline',type=Path,required=True);p.add_argument('--candidate',required=True);p.add_argument('--output',type=Path,required=True);a=p.parse_args();r=a.run;c=json.loads((r/'episode-config.json').read_text());old=pq.ParquetFile(a.baseline).read().to_pylist()[0];new=pq.ParquetFile(next((r/f'human_motion_data/{a.candidate}').rglob('motion.parquet'))).read().to_pylist()[0];objects=all(np.array_equal(old[k],new[k]) for k in ['object_body_position','object_body_wxyz']);held=True;releases=[]
for side in ['left','right']:
 grasps=sorted([g for g in c['grasps'] if g['side']==side],key=lambda g:g['start']);w0=np.array(old[f'robot_{side}_wrist_position']);w=np.array(new[f'robot_{side}_wrist_position']);q0=Rotation.from_quat(old[f'robot_{side}_wrist_wxyz'],scalar_first=True);q=Rotation.from_quat(new[f'robot_{side}_wrist_wxyz'],scalar_first=True)
 for j,g in enumerate(grasps):
  for k in ['wrist_position','wrist_wxyz','finger_joints']:held=held and np.array_equal(np.array(old[f'robot_{side}_{k}'])[g['start']:g['end']+1],np.array(new[f'robot_{side}_{k}'])[g['start']:g['end']+1])
  end=g['end'];stop=grasps[j+1]['start'] if j+1<len(grasps) else len(w)-1
  if stop<=end:continue
  before=float(np.linalg.norm(np.diff(w0[end:stop+1],axis=0),axis=1).max());after=float(np.linalg.norm(np.diff(w[end:stop+1],axis=0),axis=1).max());before_r=float((q0[end:stop].inv()*q0[end+1:stop+1]).magnitude().max());after_r=float((q[end:stop].inv()*q[end+1:stop+1]).magnitude().max());releases.append({'side':side,'release_source_frame':end,'next_grasp_source_frame':stop if j+1<len(grasps) else None,'maximum_wrist_step_before_m':before,'maximum_wrist_step_after_m':after,'maximum_rotation_step_before_rad':before_r,'maximum_rotation_step_after_rad':after_r,'continuity_improved_or_small':after<=max(.065,before*.8) and after_r<=max(.6,before_r*.9)})
report={'method':__doc__,'objects_bitwise_identical':objects,'held_wrist_and_fingers_bitwise_identical':held,'all_releases_continuous':all(x['continuity_improved_or_small'] for x in releases),'releases':releases};a.output.write_text(json.dumps(report,indent=2));print(c['episode'],report,flush=True)
