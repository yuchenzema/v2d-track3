"""Resolve estimated inter-object penetration using native collider contact normals."""
import os,json,runpy,sys,itertools,numpy as np,mujoco,pyarrow as pa,pyarrow.parquet as pq,xml.etree.ElementTree as ET
from pathlib import Path
from scipy.optimize import minimize,LinearConstraint,Bounds
import flash_chord.training.flash_sac.runner as runner
R=Path('/episode');cfg=json.loads((R/'episode-config.json').read_text());T=cfg['frames'];B=len(cfg['objects']);reference=os.environ.get('V2D_PROJECT_REFERENCE','track3_collision_variant');path=R/f"human_motion_data/{reference}/processed/sequence_id={cfg['sequence']}/robot_name=sharpa_wave/motion.parquet";row=pq.ParquetFile(path).read().to_pylist()[0]
def inspect(env,*args,**kwargs):
 e=env.env;s=e.solver;m=s.mj_model;d=s.mj_data;ids=[];addresses=[];weights=[]
 for name in cfg['objects']:
  bi=mujoco.mj_name2id(m,mujoco.mjtObj.mjOBJ_BODY,'track3_'+name+'_object');ids.append(bi);ji=int(m.body_jntadr[bi]);assert m.jnt_type[ji]==mujoco.mjtJoint.mjJNT_FREE;addresses.append(int(m.jnt_qposadr[ji]));mass=float(ET.parse(R/f'reconstruction/mesh_priors/{name}.urdf').find('.//mass').get('value'));weights.extend([max(.02,mass)]*3)
 mobile=set(json.loads(os.environ['V2D_PROJECTION_MOBILE_OBJECTS'])) if os.environ.get('V2D_PROJECTION_MOBILE_OBJECTS') else {g['object'] for g in cfg['grasps']}
 maximum=float(os.environ.get('V2D_PROJECTION_BOUND_M','.025'));assert .005<=maximum<=.05
 camera_penalty=float(os.environ.get('V2D_PROJECTION_CAMERA_PENALTY','0.'));assert 0<=camera_penalty<=1000
 cameras=np.load(R/'reconstruction/scene_motion_physical.npz')['camera_to_world'] if camera_penalty else None
 lower=np.array([-maximum if b in mobile else -1e-10 for b in range(B) for _ in range(3)]);upper=-lower;deltas=np.zeros((T,B,3));records=[]
 def contacts():
  found=[]
  for c in d.contact:
   bodies=[int(m.geom_bodyid[g]) for g in c.geom]
   if not any(bi in ids for bi in bodies):continue
   other=[mujoco.mj_id2name(m,mujoco.mjtObj.mjOBJ_BODY,bi) or 'world' for bi in bodies if bi not in ids]
   if any('sharpa_wave' in n for n in other):continue
   if c.dist>=-.0005:continue
   normal=c.frame[:3];a=np.zeros(B*3)
   for sign,bi in zip([-1,1],bodies):
    if bi in ids:b=ids.index(bi);a[b*3:b*3+3]+=sign*normal
   found.append((a,-float(c.dist)+.0005,bodies))
  return found
 for i in range(T):
  e.reset(frame_id=i);s._update_mjc_data(d,e.model,e.state_0);mujoco.mj_forward(m,d);q0=d.qpos.copy();total=np.zeros(B*3);initial=contacts();before=max([x[1]-.0005 for x in initial],default=0.)
  metric=np.diag(weights)
  if camera_penalty:
   for b,adr in enumerate(addresses):
    ray=q0[adr:adr+3]-cameras[i,:3,3];ray/=np.linalg.norm(ray)
    metric[b*3:b*3+3,b*3:b*3+3]+=weights[b*3]*camera_penalty*(np.eye(3)-np.outer(ray,ray))
  for iteration in range(8):
   hits=contacts()
   if not hits:break
   A=np.array([a for a,depth,bodies in hits]);bound=np.array([depth+a@total for a,depth,bodies in hits]);result=minimize(lambda x:float(x@metric@x),total,jac=lambda x:2*metric@x,method='SLSQP',bounds=Bounds(lower,upper),constraints=[LinearConstraint(A,bound,np.inf)],options={'ftol':1e-12,'maxiter':80})
   total=result.x
   d.qpos[:]=q0
   for b,adr in enumerate(addresses):d.qpos[adr:adr+3]+=total[b*3:b*3+3]
   mujoco.mj_forward(m,d)
  after=max([x[1]-.0005 for x in contacts()],default=0.)
  if after>.002 and mobile:
   # Overlapping convex parts can produce contradictory local normals. Try
   # bounded translations to escape that local contact configuration.
   choice=(after,float(total@metric@total),total.copy())
   for b in sorted(mobile,key=lambda b:weights[b*3])[:1]:
    for direction in itertools.product([-1,0,1],repeat=3):
     direction=np.array(direction,float)
     if not direction.any():continue
     direction/=np.linalg.norm(direction)
     for distance in [.012,.020,.025]:
      trial=np.zeros(B*3);trial[b*3:b*3+3]=direction*distance;d.qpos[:]=q0
      for j,adr in enumerate(addresses):d.qpos[adr:adr+3]+=trial[j*3:j*3+3]
      mujoco.mj_forward(m,d);depth=max([x[1]-.0005 for x in contacts()],default=0.);candidate=(max(0.,depth-.0005),float(trial@metric@trial),trial)
      if candidate[:2]<choice[:2]:choice=candidate
   total=choice[2];d.qpos[:]=q0
   for b,adr in enumerate(addresses):d.qpos[adr:adr+3]+=total[b*3:b*3+3]
   mujoco.mj_forward(m,d);after=max([x[1]-.0005 for x in contacts()],default=0.)
  if after>before+1e-6:total[:]=0.;after=before
  deltas[i]=total.reshape(B,3)
  if before>.0005:records.append({'frame':i,'penetration_before_m':before,'penetration_after_m':after,'object_translations_m':deltas[i].tolist()})
 P=np.array(row['object_body_position']);P+=deltas;row['object_body_position']=P.tolist()
 for side in ['left','right']:
  w=np.array(row[f'robot_{side}_wrist_position']);f=np.array(row[f'robot_{side}_frames']);cp=np.array(row[f'mano_{side}_object_contact_positions']);pid=np.array(row[f'mano_{side}_object_contact_part_ids'])
  for i in range(T):
   active=[g for g in cfg['grasps'] if g['side']==side and g['start']<=i<=g['end']]
   if active:delta=deltas[i,active[-1]['object']];w[i]+=delta;f[i,:,:3]+=delta
   for b in range(B):cp[i,pid[i]==b+1]+=deltas[i,b]
  row[f'robot_{side}_wrist_position']=w.tolist();row[f'robot_{side}_frames']=f.tolist();row[f'mano_{side}_object_contact_positions']=cp.tolist()
 pq.write_table(pa.Table.from_pylist([row]),path,compression='zstd');report={'method':__doc__,'maximum_translation_m':float(np.linalg.norm(deltas,axis=2).max()),'frames_adjusted':int(np.any(deltas!=0,axis=(1,2)).sum()),'raw_video_reconstruction_unchanged':True,'frames':records};(R/'research/native-object-contact-projection.json').write_text(json.dumps(report,indent=2));print(json.dumps({k:v for k,v in report.items() if k!='frames'},indent=2));raise SystemExit(0)
runner.run_training=inspect
sys.argv=[sys.argv[0],'experiment=sharpa_flash_sac',f"task.parquet='{path.parent}'",'task.motion_speed=1.0','reset.reset_finger_openness=0.0','scene.world_count=1','logging.mode=disabled','output_dir=/episode/diagnostic-checkpoints','hydra.run.dir=/episode/hydra/object-contact-projection'];runpy.run_path('/workspace/scripts/train_flash_sac.py',run_name='__main__')
