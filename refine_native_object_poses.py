"""Fit physically feasible rigid object poses to visible source pixels before learning."""
import itertools,json,os,runpy,sys
from pathlib import Path
import numpy as np,mujoco,pyarrow as pa,pyarrow.parquet as pq,trimesh
from PIL import Image,ImageDraw
from scipy.ndimage import distance_transform_edt,map_coordinates
from scipy.optimize import least_squares
from scipy.spatial import cKDTree
from scipy.spatial.transform import Rotation
import flash_chord.training.flash_sac.runner as runner

R=Path('/episode');cfg=json.loads((R/'episode-config.json').read_text());T=cfg['frames'];names=cfg['objects'];B=len(names)
reference=os.environ.get('V2D_OBJECT_POSE_REFERENCE','track3_collision_variant');path=next((R/f'human_motion_data/{reference}/processed').rglob('motion.parquet'));row=pq.ParquetFile(path).read().to_pylist()[0]
P0=np.array(row['object_body_position']);Q0=np.array(row['object_body_wxyz']);P=P0.copy();Q=Q0.copy()
raw=np.load(R/'reconstruction/scene_motion_physical.npz');measure=np.load(R/'reconstruction/hand_measurements.npz');intr=json.loads((R/'reconstruction/camera_intrinsics.json').read_text())
meshes=[trimesh.load(R/f'reconstruction/mesh_priors/{n}.ply',process=False) for n in names];centers=[m.vertices.mean(0) for m in meshes]
samples=[np.asarray(m.vertices)[np.linspace(0,len(m.vertices)-1,min(1200,len(m.vertices))).astype(int)] for m in meshes]
mobile=sorted({g['object'] for g in cfg['grasps']});static_reports=[]
if os.environ.get('V2D_OBJECT_POSE_MOBILE_OBJECTS'):mobile=list(map(int,json.loads(os.environ['V2D_OBJECT_POSE_MOBILE_OBJECTS'])))
for b in ([] if os.environ.get('V2D_PRESERVE_STATIC_POSES')=='1' else cfg.get('observed_stationary_objects',[])):
 assert b not in mobile
 rot=Rotation.from_quat(Q[:,b],scalar_first=True).mean().as_matrix();v=np.asarray(meshes[b].vertices)
 # The released rack scan's shortest extent is its height. Fit its base plane.
 axis=int(np.argmin(np.ptp(v,axis=0)));bottom=v[v[:,axis]<=np.percentile(v[:,axis],6)];_,_,basis=np.linalg.svd(bottom-bottom.mean(0),full_matrices=False);normal=basis[-1]
 if normal[axis]<0:normal=-normal
 alignment,_=Rotation.align_vectors([[0.,0.,1.]],[rot@normal]);rot=alignment.as_matrix()@rot
 center=np.median(np.einsum('tij,j->ti',Rotation.from_quat(Q[:,b],scalar_first=True).as_matrix(),centers[b])+P[:,b],axis=0);position=center-rot@centers[b];position[2]=.6505-(v@rot.T)[:,2].min();P[:,b]=position;Q[:,b]=Rotation.from_matrix(rot).as_quat(scalar_first=True)
 static_reports.append({'object':names[b],'world_pose_constant':True,'base_plane_rms_m':float(np.sqrt(np.mean(((bottom-bottom.mean(0))@normal)**2))),'maximum_translation_change_m':float(np.linalg.norm(P[:,b]-P0[:,b],axis=1).max())})

def mask(b,i):
 return np.asarray(Image.open(R/f'reconstruction/masks/{b+1}/{i:06d}.png').resize((512,384),Image.Resampling.NEAREST))>0

def inspect(env,*args,**kwargs):
 e=env.env;s=e.solver;m=s.mj_model;d=s.mj_data;ids=[];addresses=[]
 for name in names:
  bi=mujoco.mj_name2id(m,mujoco.mjtObj.mjOBJ_BODY,'track3_'+name+'_object');ids.append(bi);ji=int(m.body_jntadr[bi]);addresses.append(int(m.jnt_qposadr[ji]))
 # Keep a fixed residual dimension as the native contact set changes.
 pairs=[]
 for a in range(m.ngeom):
  ba=int(m.geom_bodyid[a])
  for b in range(a+1,m.ngeom):
   bb=int(m.geom_bodyid[b])
   if ba==bb or not (ba in [ids[j] for j in mobile] or bb in [ids[j] for j in mobile]):continue
   other=[mujoco.mj_id2name(m,mujoco.mjtObj.mjOBJ_BODY,v) or 'world' for v in [ba,bb] if v not in ids]
   if any('sharpa_wave' in name for name in other):continue
   pairs.append((a,b))
 pair_ids={pair:i for i,pair in enumerate(pairs)};previous=np.zeros(6*len(mobile));records=[]
 # A fine hollow mesh may contain hundreds of convex pieces. Grouping by
 # mobile geometry retains the maximum penetration for every such piece,
 # without allocating a mostly empty residual for every possible pair.
 compact=os.environ.get('V2D_OBJECT_POSE_COMPACT_CONTACTS')=='1'
 mobile_geoms=np.flatnonzero(np.isin(m.geom_bodyid,[ids[j] for j in mobile]))
 geom_index={int(g):i for i,g in enumerate(mobile_geoms)}
 def penetration():
  values=np.zeros(len(mobile_geoms) if compact else len(pairs))
  for c in d.contact:
   pair=tuple(sorted(map(int,c.geom)))
   if pair not in pair_ids:continue
   indices=[geom_index[g] for g in pair if g in geom_index] if compact else [pair_ids[pair]]
   for index in indices:values[index]=max(values[index],-float(c.dist)-.0005)
  return np.maximum(0.,values)
 for i in range(T):
  if i<int(os.environ.get('V2D_OBJECT_POSE_START_FRAME','0')) or i>int(os.environ.get('V2D_OBJECT_POSE_END_FRAME',str(T-1))):continue
  e.reset(frame_id=i);s._update_mjc_data(d,e.model,e.state_0)
  for b,adr in enumerate(addresses):d.qpos[adr:adr+3]=P[i,b];d.qpos[adr+3:adr+7]=Q[i,b]
  mujoco.mj_forward(m,d);q0=d.qpos.copy();initial=penetration();before=float(initial.max(initial=0.))
  if before<.002:previous*=.5;continue
  C=np.linalg.inv(raw['camera_to_world'][i]);image_constraints={}
  occlusion=Image.new('L',(512,384));draw=ImageDraw.Draw(occlusion)
  for side in range(2):
   if not measure['observed'][i,side]:continue
   uv=measure['uv'][i,side]*.5;draw.polygon([tuple(v) for v in uv[[0,1,5,9,13,17]]],fill=255)
   for finger in range(5):draw.line([tuple(v) for v in uv[[0,*range(1+4*finger,5+4*finger)]]],fill=255,width=7)
  for b in mobile:
   target=mask(b,i);visible=np.asarray(occlusion)==0
   for other in range(B):
    if other!=b:visible&=~mask(other,i)
   yy,xx=np.where(target&visible);take=np.linspace(0,max(0,len(xx)-1),min(70,len(xx))).astype(int)
   image_constraints[b]=(distance_transform_edt(~target),visible,np.c_[xx[take],yy[take]].astype(float))
  rotations=[Rotation.from_quat(Q[i,b],scalar_first=True) for b in mobile];worldcenters=[rotations[k].apply(centers[b])+P[i,b] for k,b in enumerate(mobile)]
  def apply(x):
   d.qpos[:]=q0
   for k,b in enumerate(mobile):
    adr=addresses[b];rot=Rotation.from_rotvec(x[k*6+3:k*6+6])*rotations[k];d.qpos[adr:adr+3]=worldcenters[k]+x[k*6:k*6+3]-rot.apply(centers[b]);d.qpos[adr+3:adr+7]=rot.as_quat(scalar_first=True)
   mujoco.mj_forward(m,d)
  def residual(x):
   apply(x);result=[penetration()/float(os.environ.get('V2D_OBJECT_POSE_CONTACT_SIGMA','.00045'))]
   for k,b in enumerate(mobile):
    adr=addresses[b];rot=Rotation.from_quat(d.qpos[adr+3:adr+7],scalar_first=True);v=rot.apply(samples[b])+d.qpos[adr:adr+3];camera=v@C[:3,:3].T+C[:3,3];uv=(camera[:,:2]/camera[:,2:]*[intr['fx'],intr['fy']]+[intr['cx'],intr['cy']])*.5;clipped=np.c_[np.clip(uv[:,0],0,511),np.clip(uv[:,1],0,383)];dist,visible,pixels=image_constraints[b];valid=map_coordinates(visible.astype(float),clipped[:,::-1].T,order=0,mode='nearest');forward=(map_coordinates(dist,clipped[:,::-1].T,order=1,mode='nearest')+np.linalg.norm(uv-clipped,axis=1))*valid/2/np.sqrt(len(uv)/35);result.append(forward)
    if len(pixels):result.append(cKDTree(uv).query(pixels)[0]/2/np.sqrt(len(pixels)/20))
    result.extend([x[k*6:k*6+3]/.025,x[k*6+3:k*6+6]/.3,(x[k*6:k*6+3]-previous[k*6:k*6+3])/.03,(x[k*6+3:k*6+6]-previous[k*6+3:k*6+6])/.4])
    if os.environ.get('V2D_OBJECT_POSE_TEMPORAL_GUARD')=='1':
     current_center=rot.apply(centers[b])+d.qpos[adr:adr+3]
     for neighbor in [i-1,i+1]:
      if not 0<=neighbor<T:continue
      neighbor_rot=Rotation.from_quat(Q[neighbor,b],scalar_first=True);neighbor_center=neighbor_rot.apply(centers[b])+P[neighbor,b];angle=float((neighbor_rot.inv()*rot).magnitude());distance=float(np.linalg.norm(current_center-neighbor_center));result.append(np.array([max(0.,distance-.065)/.004,max(0.,angle-.65)/.035]))
   return np.concatenate(result)
  translation_limit=float(os.environ.get('V2D_OBJECT_POSE_MAX_TRANSLATION','.04'));rotation_limit=float(os.environ.get('V2D_OBJECT_POSE_MAX_ROTATION','.45'));translation_axes=json.loads(os.environ.get('V2D_OBJECT_POSE_TRANSLATION_AXES',json.dumps([translation_limit]*3)));assert len(translation_axes)==3 and min(translation_axes)>0;lower=np.tile([-float(v) for v in translation_axes]+[-rotation_limit]*3,len(mobile));upper=-lower
  seeds=[np.zeros(len(lower)),np.clip(previous,lower+.00001,upper-.00001)]
  if os.environ.get('V2D_OBJECT_POSE_UPWARD_SEEDS')=='1':
   for height in [.005,.015,.03]:
    seed=np.zeros(len(lower));seed[2::6]=height;seeds.append(seed)
  best=None
  for seed_index in range(8):
   if seed_index>=len(seeds):
    seed=np.zeros(len(lower));axis=(seed_index-2)//2;seed[3+axis]=.28*(1 if seed_index%2 else -1)
   else:seed=seeds[seed_index]
   seed=np.clip(seed,lower+.000001,upper-.000001);fit=least_squares(residual,seed,bounds=(lower,upper),max_nfev=70,ftol=1e-5,xtol=1e-5,gtol=1e-5);apply(fit.x);after=float(penetration().max(initial=0.));cost=float(np.sum(residual(fit.x)**2));choice=(max(0.,after-.0015),cost,fit.x.copy())
   if best is None or choice[:2]<best[:2]:best=choice
   if after<.0015:break
  x=best[2];apply(x);after=float(penetration().max(initial=0.))
  if after>before:x[:]=0.;apply(x);after=before
  previous=x.copy()
  for b,adr in enumerate(addresses):P[i,b]=d.qpos[adr:adr+3];Q[i,b]=d.qpos[adr+3:adr+7]
  records.append({'frame':i,'maximum_penetration_before_m':before,'maximum_penetration_after_m':after,'correction_parameters':x.tolist()})
  if i%20==0:print(i,before,after,flush=True)
 row['object_body_position']=P.tolist();row['object_body_wxyz']=Q.tolist()
 old_rot=Rotation.from_quat(Q0.reshape(-1,4),scalar_first=True).as_matrix().reshape(T,B,3,3);new_rot=Rotation.from_quat(Q.reshape(-1,4),scalar_first=True).as_matrix().reshape(T,B,3,3);D=new_rot@old_rot.transpose(0,1,3,2);dt=P-np.einsum('tbij,tbj->tbi',D,P0)
 for side in ['left','right']:
  w=np.array(row[f'robot_{side}_wrist_position']);wq=np.array(row[f'robot_{side}_wrist_wxyz']);frames=np.array(row[f'robot_{side}_frames']);cp=np.array(row[f'mano_{side}_object_contact_positions']);cn=np.array(row[f'mano_{side}_object_contact_normals']);pid=np.array(row[f'mano_{side}_object_contact_part_ids'])
  for i in range(T):
   active=[g for g in cfg['grasps'] if g['side']==side and g['start']<=i<=g['end']]
   if active:
    b=active[-1]['object'];dr=D[i,b];shift=dt[i,b];w[i]=dr@w[i]+shift;wq[i]=(Rotation.from_matrix(dr)*Rotation.from_quat(wq[i],scalar_first=True)).as_quat(scalar_first=True);frames[i,:,:3]=frames[i,:,:3]@dr.T+shift;frames[i,:,3:]=(Rotation.from_matrix(dr)*Rotation.from_quat(frames[i,:,3:],scalar_first=True)).as_quat(scalar_first=True)
   for b in range(B):hit=pid[i]==b+1;cp[i,hit]=cp[i,hit]@D[i,b].T+dt[i,b];cn[i,hit]=cn[i,hit]@D[i,b].T
  for key,value in [('wrist_position',w),('wrist_wxyz',wq),('frames',frames)]:row[f'robot_{side}_{key}']=value.tolist()
  row[f'mano_{side}_object_contact_positions']=cp.tolist();row[f'mano_{side}_object_contact_normals']=cn.tolist()
 pq.write_table(pa.Table.from_pylist([row]),path,compression='zstd');report={'method':__doc__,'reference':reference,'static_source_observations':static_reports,'maximum_position_change_m':float(np.linalg.norm(P-P0,axis=-1).max()),'maximum_orientation_change_rad':float(Rotation.from_matrix(D.reshape(-1,3,3)).magnitude().max()),'raw_source_reconstruction_preserved':True,'frames':records};(R/os.environ.get('V2D_OBJECT_POSE_OUTPUT','research/native-object-6d-refinement.json')).write_text(json.dumps(report,indent=2));print({k:v for k,v in report.items() if k!='frames'},flush=True);raise SystemExit(0)
runner.run_training=inspect
sys.argv=[sys.argv[0],'experiment=sharpa_flash_sac',f"task.parquet='{path.parent}'",'task.motion_speed=1.0','reset.reset_finger_openness=0.0','scene.world_count=1','logging.mode=disabled','output_dir=/episode/diagnostic-checkpoints','hydra.run.dir=/episode/hydra/object-6d-refinement'];runpy.run_path('/workspace/scripts/train_flash_sac.py',run_name='__main__')
