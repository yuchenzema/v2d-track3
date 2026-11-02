"""Find source-consistent supported poses using continuous, unassisted Newton tests.

This is reconstruction calibration, never a learned-policy rollout. Each independent
candidate is initialized once, then evolves under gravity for three physical seconds.
"""
import json,os,runpy,sys
from pathlib import Path
import hydra,mujoco,newton,numpy as np,trimesh,warp as wp
from PIL import Image
from scipy.ndimage import distance_transform_edt,map_coordinates
from scipy.spatial import cKDTree
from scipy.spatial.transform import Rotation
from omegaconf import OmegaConf
import flash_chord.training.flash_sac.runner as runner
from flash_chord.evaluation.harness import apply_evaluation_composition

r=Path('/episode');cfg=json.loads((r/'episode-config.json').read_text());ep=cfg['episode'];W=32
b=int(os.environ.get('V2D_SUPPORT_OBJECT',1 if ep in [16,28] else 0));name=cfg['objects'][b]
reference=cfg['training_reference'];substeps=int(cfg.get('physics_substeps',5));raw=np.load(r/'reconstruction/scene_motion_physical.npz')
intr=json.loads((r/'reconstruction/camera_intrinsics.json').read_text());last=int(os.environ.get('V2D_SUPPORT_SOURCE_FRAME',cfg['frames']-1));assert 0<=last<cfg['frames'];C=np.linalg.inv(raw['camera_to_world'][last])
label=os.environ.get('V2D_SUPPORT_SEARCH_LABEL','terminal-support-search');assert label.replace('-','').isalnum()
mesh=trimesh.load(r/f'reconstruction/mesh_priors/{name}.ply',process=False);v=np.array(mesh.vertices);sample=v[np.linspace(0,len(v)-1,min(2000,len(v))).astype(int)];center=v.mean(0)
target=np.array(Image.open(r/f'reconstruction/masks/{b+1}/{last:06d}.png').resize((512,384),Image.Resampling.NEAREST))>0
distance=distance_transform_edt(~target);yy,xx=np.where(target);ids=np.linspace(0,len(xx)-1,min(200,len(xx))).astype(int);pixels=np.c_[xx[ids],yy[ids]]

def image_cost(poses):
 result=[]
 for pose in poses:
  world=Rotation.from_quat(pose[3:]).apply(sample)+pose[:3];pc=world@C[:3,:3].T+C[:3,3];uv=(pc[:,:2]/pc[:,2:]*[intr['fx'],intr['fy']]+[intr['cx'],intr['cy']])*.5
  clipped=np.clip(uv,[0,0],[511,383]);dist=map_coordinates(distance,clipped[:,::-1].T,order=1,mode='nearest')+np.linalg.norm(uv-clipped,axis=1)
  result.append(float(np.mean(np.minimum(dist,50)**2)+np.mean(np.minimum(cKDTree(uv).query(pixels)[0],50)**2)))
 return np.array(result)

def inspect(environment,*args,**kwargs):
 e=environment.env;assert not e.config.auto_reset;e.set_voc_scale(0.);reference_last=round(last*(e.reference.num_frames-1)/(cfg['frames']-1));e.reset(frame_id=reference_last)
 binding=e.scene.objects[b];qid=np.array(binding.root.free_q_ids);stride=e.model.joint_coord_count//W;body_stride=e.model.body_count//W;bodyid=binding.root.body_id
 mm=e.solver.mj_model;dd=e.solver.mj_data;object_ids=[];addresses=[]
 for objname in cfg['objects']:
  bi=mujoco.mj_name2id(mm,mujoco.mjtObj.mjOBJ_BODY,'track3_'+objname+'_object');assert bi>=0,objname;object_ids.append(bi);addresses.append(int(mm.jnt_qposadr[int(mm.body_jntadr[bi])]))
 reference_p=e.reference.object_body_pos_w()[reference_last];reference_q=e.reference.object_body_quat_w()[reference_last]
 for j,adr in enumerate(addresses):dd.qpos[adr:adr+3]=reference_p[j];dd.qpos[adr+3:adr+7]=reference_q[j]
 base_qpos=dd.qpos.copy()
 def static_penetration(poses):
  values=[]
  for pose in poses:
   dd.qpos[:]=base_qpos;adr=addresses[b];dd.qpos[adr:adr+3]=pose[:3];dd.qpos[adr+3:adr+7]=pose[[6,3,4,5]];mujoco.mj_forward(mm,dd);maximum=0.
   for contact in dd.contact:
    bodies=[int(mm.geom_bodyid[g]) for g in contact.geom]
    if object_ids[b] not in bodies:continue
    labels=[mujoco.mj_id2name(mm,mujoco.mjtObj.mjOBJ_BODY,body) or 'world' for body in bodies]
    if any('sharpa_wave' in label for label in labels):continue
    maximum=max(maximum,-float(contact.dist))
   values.append(maximum)
  return np.array(values)
 initial=e.state_0.body_q.numpy()[bodyid].copy();R0=Rotation.from_quat(initial[3:]);center0=R0.apply(center)+initial[:3];rng=np.random.default_rng(20261016+ep)
 means=np.zeros(6);scale=np.array([.025,.025,.015,.22,.22,.18]) if ep!=16 else np.array([.04,.06,.045,.5,.5,.3]);archive=[];best=None
 for iteration in range(8):
  x=rng.normal(means,scale,size=(W,6));x[0]=means;x[1]=0
  if iteration==0 and ep==16:
   direction=R0.apply([0,0,1]);direction*=1 if direction[2]>0 else -1
   upright,_=Rotation.align_vectors([[0,0,1]],[direction]);rv=upright.as_rotvec()
   for j,f in enumerate([.4,.6,.8,1.]):
    for k,height in enumerate([.03,.065,.09]):
     index=2+j*3+k;x[index]=0;x[index,2]=height;x[index,3:]=f*rv
  bound=.14 if ep==16 else .065;x[:,:3]=np.clip(x[:,:3],-bound,bound);x[:,3:]=np.clip(x[:,3:],-1.2 if ep==16 else -.65,1.2 if ep==16 else .65)
  rot=Rotation.from_rotvec(x[:,3:])*R0;position=center0+x[:,:3]-rot.apply(np.broadcast_to(center,(W,3)).copy())
  e.reset(frame_id=reference_last);q=e.state_0.joint_q.numpy().reshape(W,stride);q[:,qid[:3]]=position;q[:,qid[3:]]=rot.as_quat();e.state_0.joint_q.assign(q.ravel().astype(np.float32));newton.eval_fk(e.model,e.state_0.joint_q,e.state_0.joint_qd,e.state_0)
  poses=[];action=wp.zeros_like(e.action_input)
  for step in range(60):
   e.timestep.fill_(reference_last);e.step(action)
   if step>=49:poses.append(e.state_0.body_q.numpy()[np.arange(W)*body_stride+bodyid])
  poses=np.array(poses);final=poses[-1];linear=np.linalg.norm(poses[:,:,:3]-final[None,:,:3],axis=-1).max(0);angular=(Rotation.from_quat(np.broadcast_to(final[None,:,3:],poses[:,:,3:].shape).reshape(-1,4).copy()).inv()*Rotation.from_quat(poses[:,:,3:].reshape(-1,4))).magnitude().reshape(len(poses),W).max(0)
  delta=Rotation.from_quat(final[:,3:])*R0.inv();angle=delta.magnitude();center_shift=np.linalg.norm(Rotation.from_quat(final[:,3:]).apply(np.broadcast_to(center,(W,3)).copy())+final[:,:3]-center0,axis=1)
  photo=image_cost(final);physical=(linear/.002)**2+(angular/.03)**2;plausible=(center_shift<(.16 if ep==16 else .09))&(angle<(1.5 if ep==16 else .85))&np.isfinite(final).all(1)
  # A pose must also survive a fresh initialization against the original
  # rack, rather than balancing only after the rack has moved during settling.
  e.reset(frame_id=reference_last);q=e.state_0.joint_q.numpy().reshape(W,stride);q[:,qid[:3]]=final[:,:3];q[:,qid[3:]]=final[:,3:];e.state_0.joint_q.assign(q.ravel().astype(np.float32));newton.eval_fk(e.model,e.state_0.joint_q,e.state_0.joint_qd,e.state_0)
  restarted=[]
  for step in range(60):
   e.timestep.fill_(reference_last);e.step(action);restarted.append(e.state_0.body_q.numpy()[np.arange(W)*body_stride+bodyid])
  restarted=np.array(restarted);restart_position=np.linalg.norm(restarted[:,:,:3]-final[None,:,:3],axis=-1).max(0);restart_rotation=(Rotation.from_quat(np.broadcast_to(final[None,:,3:],restarted[:,:,3:].shape).reshape(-1,4).copy()).inv()*Rotation.from_quat(restarted[:,:,3:].reshape(-1,4))).magnitude().reshape(len(restarted),W).max(0)
  penetration=static_penetration(final)
  score=photo+2*physical+(center_shift/.05)**2+(angle/.65)**2+2*(restart_position/.01)**2+(restart_rotation/.15)**2+5*(np.maximum(0.,penetration-.003)/.002)**2;score[~plausible]+=1e6;stable=plausible&(linear<.003)&(angular<.04)&(restart_position<.012)&(restart_rotation<.15)&(penetration<.008)
  for j in range(W):
   row={'round':iteration,'candidate':j,'parameters':x[j].tolist(),'initial_pose':np.r_[position[j],rot[j].as_quat()].tolist(),'settled_pose':final[j].tolist(),'image_chamfer_cost':float(photo[j]),'residual_translation_m':float(linear[j]),'residual_rotation_rad':float(angular[j]),'restart_maximum_translation_m':float(restart_position[j]),'restart_maximum_rotation_rad':float(restart_rotation[j]),'native_static_object_penetration_m':float(penetration[j]),'stable_and_plausible':bool(stable[j]),'score':float(score[j])};archive.append(row)
   if stable[j] and (best is None or row['score']<best['score']):best=row
  order=np.argsort(score)[:8];means=x[order].mean(0);scale=np.maximum(x[order].std(0),np.array([.004,.004,.003,.035,.035,.035]))
  print('SUPPORT_SEARCH',ep,name,iteration,'stable',int(stable.sum()),'best',None if best is None else {k:best[k] for k in ['score','image_chamfer_cost','residual_translation_m','residual_rotation_rad']},flush=True)
  report={'method':__doc__,'search_version':3,'object':name,'object_index':b,'reference':reference,'object_assistance':0.,'policy_inference':False,'physics_integration_hz':20*substeps,'physical_candidate_duration_s':3.,'independent_restart_duration_s':3.,'reference_final_frame':reference_last,'source_frame':last,'initial_reference_pose':initial.tolist(),'baseline_image_chamfer_cost':float(image_cost([initial])[0]),'best':best,'candidates':archive};(r/f'research/{label}.json').write_text(json.dumps(report,indent=2))
 raise SystemExit(0)

runner.run_training=inspect;sys.argv=[sys.argv[0],'experiment=sharpa_flash_sac',f"task.parquet='/episode/human_motion_data/{reference}/processed/sequence_id={cfg['sequence']}/robot_name=sharpa_wave'",'task.motion_speed=0.5',f'sim.substeps={substeps}','sim.njmax=1024','sim.nconmax=512','reset.reset_finger_openness=0.0',f'scene.world_count={W}','logging.mode=disabled','output_dir=/episode/diagnostic-checkpoints','hydra.run.dir=/episode/hydra/terminal-support-search'];native=runpy.run_path('/workspace/scripts/train_flash_sac.py',run_name='native_training')
@hydra.main(version_base='1.3',config_path='/workspace/src/flash_chord/configs',config_name='train_flash_sac')
def main(config):
 config=OmegaConf.create(apply_evaluation_composition(OmegaConf.to_container(config,resolve=True),world_count=W,reset_mode='explicit'));native['main'].__wrapped__(config)
main()
