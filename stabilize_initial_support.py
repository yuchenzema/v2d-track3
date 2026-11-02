"""Fit source-observed initial rests to stable native support geometry before learning."""
from pathlib import Path
import argparse,json,xml.etree.ElementTree as ET
import numpy as np,pyarrow as pa,pyarrow.parquet as pq,trimesh
from scipy.optimize import minimize
from scipy.spatial.transform import Rotation,Slerp
from flat_tool_rests import fit

p=argparse.ArgumentParser();p.add_argument('--run',type=Path,required=True);p.add_argument('--reference');p.add_argument('--allow-frozen-alternate',action='store_true');a=p.parse_args();r=a.run.resolve();cfg=json.loads((r/'episode-config.json').read_text());reference=a.reference or cfg.get('training_reference','track3_coherent_variant')
if (r/'coherent-input-checksums.json').exists():assert a.allow_frozen_alternate and reference!=cfg.get('training_reference','track3_coherent_variant')
path=next((r/f'human_motion_data/{reference}/processed').rglob('motion.parquet'));row=pq.ParquetFile(path).read().to_pylist()[0];P0=np.array(row['object_body_position']);Q0=np.array(row['object_body_wxyz']);P=P0.copy();Q=Q0.copy();T=cfg['frames'];B=len(cfg['objects']);raw=np.load(r/'reconstruction/scene_motion_physical.npz');hulls=np.load(r/'evaluation/object-collision-hulls.npz');records=[]
eligible={14:['blue_cup'],15:['wooden_spoon'],16:['wooden_spoon'],19:['mini_sweeper','mini_dust_pan'],10:['blue_cup','beige_cup']}.get(cfg['episode'],[])
for b,name in enumerate(cfg['objects']):
 if name not in eligible:continue
 start=min(g['start'] for g in cfg['grasps'] if g['object']==b)
 if start==0:continue
 end=start-1;anchor_end=min(end,15);mesh=trimesh.load(r/f'reconstruction/mesh_priors/{name}.ply',process=False);vertices=np.asarray(mesh.vertices);center=vertices.mean(0);native=np.concatenate([hulls[k] for k in hulls.files if k.startswith(name+'__')]);initial=Rotation.from_quat(Q[:anchor_end+1,b],scalar_first=True).as_matrix();R0=Rotation.from_matrix(initial).mean().as_matrix();extra={};energy=None
 if name!='wooden_spoon':
  origin=ET.parse(r/f'reconstruction/mesh_priors/{name}.urdf').find('.//inertial/origin');com=np.fromstring(origin.get('xyz','0 0 0'),sep=' ') if origin is not None else np.zeros(3)
  def potential(x):
   rot=Rotation.from_rotvec([x[0],x[1],0.]).as_matrix()@R0;return float(-np.min((native@rot.T)[:,2])+(rot@com)[2]+1e-5*(x@x))
  optimum=minimize(potential,np.zeros(2),method='Powell',bounds=[(-.2,.2),(-.2,.2)],options={'xtol':1e-8,'ftol':1e-10,'maxiter':100});stable=Rotation.from_rotvec([*optimum.x,0.]).as_matrix()@R0;extra={'fixed_rotation':stable,'support_vertices':native};energy={'tilt_correction_rad':optimum.x.tolist(),'potential_height_before_m':potential(np.zeros(2)),'potential_height_after_m':potential(optimum.x),'center_of_mass_from_urdf':com.tolist(),'optimizer_success':bool(optimum.success)}
 ids=np.unique(np.linspace(0,anchor_end,min(7,anchor_end+1)).round().astype(int));model,photometric=fit(r,vertices,center,raw['camera_to_world'],ids,b,initial,**extra);P[:end+1,b]=model[:3,3];Q[:end+1,b]=Rotation.from_matrix(model[:3,:3]).as_quat(scalar_first=True)
 for i in range(end+1,min(T,end+5)):
  blend=(end+5-i)/5;old=np.eye(4);old[:3,:3]=Rotation.from_quat(Q0[i,b],scalar_first=True).as_matrix();old[:3,3]=P0[i,b];old_center=old[:3,:3]@center+old[:3,3];target_center=model[:3,:3]@center+model[:3,3];rot=Slerp([0,1],Rotation.from_matrix([old[:3,:3],model[:3,:3]]))([blend])[0].as_matrix();Q[i,b]=Rotation.from_matrix(rot).as_quat(scalar_first=True);P[i,b]=(1-blend)*old_center+blend*target_center-rot@center;P[i,b,2]+=max(0.,.6505-(native@rot.T+P[i,b])[:,2].min())
 records.append({'object':name,'initial_rest_interval':[0,end],'energy_minimization':energy,'source_image_fit':photometric})
old_rot=Rotation.from_quat(Q0.reshape(-1,4),scalar_first=True).as_matrix().reshape(T,B,3,3);new_rot=Rotation.from_quat(Q.reshape(-1,4),scalar_first=True).as_matrix().reshape(T,B,3,3);D=new_rot@old_rot.transpose(0,1,3,2);dt=P-np.einsum('tbij,tbj->tbi',D,P0);row['object_body_position']=P.tolist();row['object_body_wxyz']=Q.tolist()
for side in ['left','right']:
 w=np.array(row[f'robot_{side}_wrist_position']);wq=np.array(row[f'robot_{side}_wrist_wxyz']);frames=np.array(row[f'robot_{side}_frames']);cp=np.array(row[f'mano_{side}_object_contact_positions']);cn=np.array(row[f'mano_{side}_object_contact_normals']);pid=np.array(row[f'mano_{side}_object_contact_part_ids'])
 for i in range(T):
  active=[g for g in cfg['grasps'] if g['side']==side and g['start']<=i<=g['end']]
  if active:
   b=active[-1]['object'];dr=D[i,b];shift=dt[i,b];w[i]=dr@w[i]+shift;wq[i]=(Rotation.from_matrix(dr)*Rotation.from_quat(wq[i],scalar_first=True)).as_quat(scalar_first=True);frames[i,:,:3]=frames[i,:,:3]@dr.T+shift;frames[i,:,3:]=(Rotation.from_matrix(dr)*Rotation.from_quat(frames[i,:,3:],scalar_first=True)).as_quat(scalar_first=True)
  for b in range(B):hit=pid[i]==b+1;cp[i,hit]=cp[i,hit]@D[i,b].T+dt[i,b];cn[i,hit]=cn[i,hit]@D[i,b].T
 for key,value in [('wrist_position',w),('wrist_wxyz',wq),('frames',frames)]:row[f'robot_{side}_{key}']=value.tolist()
 row[f'mano_{side}_object_contact_positions']=cp.tolist();row[f'mano_{side}_object_contact_normals']=cn.tolist()
pq.write_table(pa.Table.from_pylist([row]),path,compression='zstd');report={'method':__doc__,'reference':reference,'initial_rests':records,'maximum_position_change_m':float(np.linalg.norm(P-P0,axis=-1).max()),'raw_video_reconstruction_preserved':True,'original_frame_grid_preserved':True};(r/f'research/{reference}-initial-support-refinement.json').write_text(json.dumps(report,indent=2));print(cfg['episode'],[{k:v for k,v in x.items() if k!='source_image_fit'} for x in records],flush=True)
