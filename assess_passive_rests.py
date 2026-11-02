"""Check passive motion only before actual first robot contact, independently of policy quality."""
import argparse,json
from pathlib import Path
import numpy as np,pyarrow.parquet as pq
from scipy.spatial.transform import Rotation
p=argparse.ArgumentParser();p.add_argument('--run',type=Path,required=True);p.add_argument('--label',default='passive-rest-preflight');a=p.parse_args();r=a.run;c=json.loads((r/'episode-config.json').read_text());reference=json.loads((r/f'evaluation/{a.label}-protocol.json').read_text())['reference'];data=np.load(r/f'evaluation/{a.label}-states.npz');trajectory=data['body_poses'];names=data['body_names'].tolist();contacts=json.loads((r/f'evaluation/{a.label}-contacts.json').read_text());motion=pq.ParquetFile(next((r/f'human_motion_data/{reference}').rglob('motion.parquet'))).read().to_pylist()[0];rows=[]
for b,name in enumerate(c['objects']):
 start=min([g['start'] for g in c['grasps'] if g['object']==b],default=c['frames']);cutoff=min(len(trajectory),max(0,2*(start-3)))
 touch=next((x['control_frame'] for x in contacts if any(hit['distance_m']<0 and any(name in n for n in hit['bodies']) and (any('sharpa_wave' in n for n in hit['bodies']) or any(other!=name and any(other in n for n in hit['bodies']) for other in c['objects'])) for hit in x['contacts'])),None)
 if touch is not None:cutoff=min(cutoff,touch)
 cutoff=min(cutoff,len(contacts))
 if cutoff==0:continue
 k=names.index(f'track3_{name}/object');initial=data['initial_body_poses'][k] if 'initial_body_poses' in data else np.r_[motion['object_body_position'][0][b],np.array(motion['object_body_wxyz'][0][b])[[1,2,3,0]]];position=np.linalg.norm(trajectory[:cutoff,k,:3]-initial[:3],axis=1);angle=(Rotation.from_quat(initial[3:]).inv()*Rotation.from_quat(trajectory[:cutoff,k,3:])).magnitude();bad=(position>.01)|(angle>.2)
 rows.append({'object':name,'control_frames_before_approach_or_actual_contact':cutoff,'first_external_contact_control_frame':touch,'contact_window_excludes_hands_and_other_task_objects':True,'maximum_position_drift_m':float(position.max()),'maximum_orientation_drift_rad':float(angle.max()),'first_unstable_control_frame':int(np.flatnonzero(bad)[0]) if bad.any() else None,'passed_1cm_0_2rad_passive_rest':bool(not bad.any())})
report={'method':__doc__,'diagnostic_only':True,'policy_inference':False,'reference':reference,'objects':rows};target=r/f'evaluation/{a.label}-initial-rest.json';temporary=target.with_suffix('.host.tmp');temporary.write_text(json.dumps(report,indent=2));temporary.replace(target);print(c['episode'],report,flush=True)
