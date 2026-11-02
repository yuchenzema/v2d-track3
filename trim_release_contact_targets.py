"""End required fingertip-contact targets at the video-observed release frame."""
from pathlib import Path
import argparse,hashlib,json
import numpy as np,pyarrow as pa,pyarrow.parquet as pq
p=argparse.ArgumentParser();p.add_argument('--run',type=Path,required=True);a=p.parse_args();r=a.run.resolve();assert not (r/'coherent-input-checksums.json').exists(),'Never alter a frozen learner reference'
c=json.loads((r/'episode-config.json').read_text());reference=c.get('training_reference','track3_coherent_variant');path=next((r/f'human_motion_data/{reference}/processed').rglob('motion.parquet'));row=pq.ParquetFile(path).read().to_pylist()[0];T=c['frames'];report={};changed=False
for side in ['left','right']:
 allowed=np.zeros(T,bool)
 for grasp in c['grasps']:
  if grasp['side']==side:allowed[max(0,grasp['start']-3):grasp['end']+1]=True
 ids=np.array(row[f'mano_{side}_object_contact_part_ids']);remove=(~allowed)[:,None]&(ids>0);report[side]={'removed_fingertip_targets':int(remove.sum()),'source_frames':np.flatnonzero(remove.any(axis=1)).tolist()};changed|=bool(remove.any());ids[remove]=0;row[f'mano_{side}_object_contact_part_ids']=ids.tolist()
 for field in ['positions','normals']:
  values=np.array(row[f'mano_{side}_object_contact_{field}']);values[remove]=0.;row[f'mano_{side}_object_contact_{field}']=values.tolist()
before=hashlib.sha256(path.read_bytes()).hexdigest()
if changed:pq.write_table(pa.Table.from_pylist([row]),path,compression='zstd')
out={'method':__doc__,'reference':reference,'before_sha256':before,'after_sha256':hashlib.sha256(path.read_bytes()).hexdigest(),'object_and_robot_pose_columns_unchanged':True,'allowed_contact_approach_lead_frames':3,'post_release_contact_requirement_removed':True,'hands':report};marker=r/'research/release-contact-targets.json'
if changed or not marker.exists():marker.write_text(json.dumps(out,indent=2))
print(c['episode'],'release contact labels',report,flush=True)
