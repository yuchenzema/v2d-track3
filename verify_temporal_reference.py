"""Audit pose continuity and unchanged terminal anchors for a separate candidate."""
import argparse,json
from pathlib import Path
import numpy as np,pyarrow.parquet as pq,trimesh
from scipy.spatial.transform import Rotation
p=argparse.ArgumentParser();p.add_argument('--run',required=True,type=Path);p.add_argument('--reference',required=True);p.add_argument('--baseline',required=True);a=p.parse_args();c=json.loads((a.run/'episode-config.json').read_text())
def load(ref):return pq.ParquetFile(next((a.run/f'human_motion_data/{ref}/processed').rglob('motion.parquet'))).read().to_pylist()[0]
row=load(a.reference);old=load(a.baseline);P=np.array(row['object_body_position']);Q=np.array(row['object_body_wxyz']);rows=[]
for b,name in enumerate(c['objects']):
 center=np.asarray(trimesh.load(a.run/f'reconstruction/mesh_priors/{name}.ply',process=False).vertices).mean(0);R=Rotation.from_quat(Q[:,b],scalar_first=True);C=R.apply(np.tile(center,(len(P),1)))+P[:,b];d=np.linalg.norm(np.diff(C,axis=0),axis=1);angle=(R[:-1].inv()*R[1:]).magnitude();oldR=Rotation.from_quat(np.array(old['object_body_wxyz'])[:,b],scalar_first=True);oldC=oldR.apply(np.tile(center,(len(P),1)))+np.array(old['object_body_position'])[:,b];original_max=float(np.linalg.norm(np.diff(oldC,axis=0),axis=1).max());limit=max(.078,original_max+.003);rows.append({'object':name,'maximum_centroid_step_m':float(d.max()),'original_maximum_centroid_step_m':original_max,'centroid_step_limit_m':limit,'centroid_criterion':'Retain existing video-derived translation speeds; reject a newly introduced peak-step increase above 3 mm, with a 78 mm floor. Camera-depth uncertainty remains.','maximum_orientation_step_rad':float(angle.max()),'passed':bool(d.max()<=limit and angle.max()<=.74)})
terminal_same=np.allclose(P[-1],np.array(old['object_body_position'])[-1],atol=1e-8) and np.max((Rotation.from_quat(np.array(old['object_body_wxyz'])[-1],scalar_first=True).inv()*Rotation.from_quat(Q[-1],scalar_first=True)).magnitude())<1e-8;print(json.dumps({'passed':bool(all(x['passed'] for x in rows) and terminal_same),'objects':rows,'terminal_object_poses_unchanged':bool(terminal_same)}))
