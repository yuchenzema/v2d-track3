"""Project estimated reference out of actual table collision hulls, preserving held grasps."""
import argparse,json,re
from pathlib import Path
import numpy as np,pyarrow as pa,pyarrow.parquet as pq
from scipy.spatial.transform import Rotation
p=argparse.ArgumentParser();p.add_argument('--run',type=Path,required=True);a=p.parse_args();r=a.run;c=json.loads((r/'episode-config.json').read_text());path=next((r/'human_motion_data/track3_collision_variant/processed').rglob('motion.parquet'));row=pq.ParquetFile(path).read().to_pylist()[0];P=np.array(row['object_body_position']);Q=np.array(row['object_body_wxyz']);T,B,_=P.shape;h=np.load(r/'evaluation/object-collision-hulls.npz');delta=np.zeros((T,B,3));report=[]
support=next((r/'human_motion_data/track3_collision_variant/reconstructed_stage').glob('*support.usda')).read_text();radius=float(re.search(r'double radius = ([0-9.e+-]+)',support)[1]);height=float(re.search(r'double height = ([0-9.e+-]+)',support)[1]);center=np.array([float(x) for x in re.search(r'xformOp:translate = \(([^)]+)\)',support)[1].split(',')]);surface=center[2]+height/2+.0005
for b,name in enumerate(c['objects']):
 vertices=np.concatenate([h[k] for k in h.files if k.startswith(name+'__')]);rot=Rotation.from_quat(Q[:,b],scalar_first=True).as_matrix()
 for i in range(T):
  v=vertices@rot[i].T+P[i,b];inside=np.linalg.norm(v[:,:2]-center[:2],axis=1)<radius
  if inside.any():delta[i,b,2]=max(0.,surface-v[inside,2].min())
 assert delta[:,b,2].max()<c.get('maximum_floor_projection_m',.025),(name,float(delta[:,b,2].max()))
 report.append({'object':name,'frames_projected':int(np.count_nonzero(delta[:,b,2])),'maximum_lift_m':float(delta[:,b,2].max())})
P+=delta;row['object_body_position']=P.tolist()
for side in ['left','right']:
 wrist=np.array(row[f'robot_{side}_wrist_position']);frames=np.array(row[f'robot_{side}_frames']);contact=np.array(row[f'mano_{side}_object_contact_positions']);ids=np.array(row[f'mano_{side}_object_contact_part_ids'])
 for i in range(T):
  active=[g for g in c['grasps'] if g['side']==side and g['start']<=i<=g['end']]
  if active:
   d=delta[i,active[-1]['object']];wrist[i]+=d;frames[i,:,:3]+=d
  for b in range(B):contact[i,ids[i]==b+1]+=delta[i,b]
 row[f'robot_{side}_wrist_position']=wrist.tolist();row[f'robot_{side}_frames']=frames.tolist();row[f'mano_{side}_object_contact_positions']=contact.tolist()
pq.write_table(pa.Table.from_pylist([row]),path,compression='zstd');out={'method':__doc__,'table_radius_used_m':radius,'table_surface_with_clearance_m':surface,'raw_video_reconstruction_unchanged':True,'objects':report};(r/'research/collision-floor-projection.json').write_text(json.dumps(out,indent=2));print(json.dumps(out,indent=2))
