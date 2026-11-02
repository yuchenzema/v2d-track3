"""Audit video cup alignment and axial nesting clearance using actual collision hulls."""
import os,json,sys,runpy
from pathlib import Path
import numpy as np,mujoco,trimesh
from scipy.spatial.transform import Rotation
import flash_chord.training.flash_sac.runner as runner
r=Path('/episode');cfg=json.loads((r/'episode-config.json').read_text());ref=os.environ.get('V2D_AUDIT_REFERENCE','track3_video')
def inspect(environment,*args,**kwargs):
 e=environment.env;s=e.solver;m=s.mj_model;d=s.mj_data;ids=[];addresses=[];centers=[];axes=[]
 for n in cfg['objects']:
  bi=mujoco.mj_name2id(m,mujoco.mjtObj.mjOBJ_BODY,'track3_'+n+'_object');ids.append(bi);addresses.append(int(m.jnt_qposadr[int(m.body_jntadr[bi])]))
  mesh=trimesh.load(r/f'reconstruction/mesh_priors/{n}.ply',process=False);v=np.asarray(mesh.vertices);_,_,V=np.linalg.svd(v-v.mean(0),full_matrices=False);centers.append(v.mean(0));axes.append(V[0])
 rows=[]
 for t in [80,100,140,160]:
  e.reset(frame_id=t);s._update_mjc_data(d,e.model,e.state_0);mujoco.mj_forward(m,d);q0=d.qpos.copy();a,b=addresses
  rots=[Rotation.from_quat(q0[x+3:x+7],scalar_first=True) for x in addresses]
  ax=[R.apply(axis) for R,axis in zip(rots,axes)];ax=[v*(1 if v[2]>0 else -1) for v in ax]
  world=[R.apply(c)+q0[x:x+3] for R,c,x in zip(rots,centers,addresses)];align,_=Rotation.align_vectors([ax[0]],[ax[1]])
  candidates=[]
  for alignment in [0.,.5,1.]:
   rot=Rotation.from_rotvec(align.as_rotvec()*alignment)*rots[1]
   for center_fraction in [0.,.5,1.]:
    origin=(1-center_fraction)*world[1]+center_fraction*(world[0]+ax[0]*np.dot(world[1]-world[0],ax[0]))
    for height in np.linspace(0.,.12,61):
     d.qpos[:]=q0;d.qpos[b:b+3]=origin+ax[0]*height-rot.apply(centers[1]);d.qpos[b+3:b+7]=rot.as_quat(scalar_first=True);mujoco.mj_forward(m,d)
     depth=max([max(0.,-float(c.dist)) for c in d.contact if set(int(m.geom_bodyid[g]) for g in c.geom)==set(ids)],default=0.)
     candidates.append({'axial_translation_m':float(height),'axis_alignment_fraction':alignment,'centering_fraction':center_fraction,'maximum_object_penetration_m':depth,'pose':d.qpos[b:b+7].tolist()})
  feasible=[v for v in candidates if v['maximum_object_penetration_m']<=.0015]
  best=min(feasible,key=lambda x:x['axial_translation_m']+.01*x['axis_alignment_fraction']+.005*x['centering_fraction']) if feasible else min(candidates,key=lambda x:x['maximum_object_penetration_m'])
  out={'source_frame':t,'original_axis_angle_rad':float(np.arccos(np.clip(np.dot(*ax),-1,1))),'feasible_candidate_count':len(feasible),'best':best};rows.append(out);print(out,flush=True)
 (r/'research/cup-nesting-axial-clearance.json').write_text(json.dumps({'diagnostic_only':True,'reference':ref,'frames':rows},indent=2));raise SystemExit(0)
runner.run_training=inspect
path=next((r/f'human_motion_data/{ref}/processed').rglob('motion.parquet'))
sys.argv=[sys.argv[0],'experiment=sharpa_flash_sac',f"task.parquet='{path.parent}'",'task.motion_speed=1.0','reset.reset_finger_openness=0.0','scene.world_count=1','logging.mode=disabled','output_dir=/episode/diagnostic-checkpoints','hydra.run.dir=/episode/hydra/cup-nesting-axial-clearance'];runpy.run_path('/workspace/scripts/train_flash_sac.py',run_name='__main__')
