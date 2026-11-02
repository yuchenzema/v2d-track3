"""Check actual Newton/MuJoCo collision geometry at every video reference frame."""
import os,json,runpy,sys,numpy as np,mujoco,newton
from pathlib import Path
import flash_chord.training.flash_sac.runner as runner
from scipy.spatial.transform import Rotation
R=Path('/episode');cfg=json.loads((R/'episode-config.json').read_text());T=cfg['frames'];seq=cfg['sequence'];reference=os.environ.get('V2D_AUDIT_REFERENCE','track3_coherent_variant')
penetration_epsilon=float(os.environ.get('V2D_COLLISION_EPSILON_M','.001'))
assert 0<=penetration_epsilon<=.001
def inspect(env,*args,**kwargs):
 e=env.env;s=e.solver;m=s.mj_model;d=s.mj_data;rows=[];hulls={};flags=e.model.shape_flags.numpy();scale=e.model.shape_scale.numpy();shapeq=e.model.shape_transform.numpy();pose_auditor=None
 for binding in e.scene.objects:
  for k,shape in enumerate(binding.shapes.ids()):
   if not int(flags[shape])&int(newton.ShapeFlags.COLLIDE_SHAPES):continue
   mesh=e.model.shape_source[shape]
   if mesh is None:continue
   q=shapeq[shape];v=np.asarray(mesh.vertices)*scale[shape];hulls[f'{binding.name}__{k}']=v@Rotation.from_quat(q[3:]).as_matrix().T+q[:3]
 np.savez_compressed(R/'evaluation/object-collision-hulls.npz',**hulls)
 for i in range(T):
  frame=int(round(i*(e.reference.num_frames-1)/(T-1)));e.reset(frame_id=frame);s._update_mjc_data(d,e.model,e.state_0);mujoco.mj_forward(m,d);contacts=[]
  for c in d.contact:
   if c.dist>=-penetration_epsilon:continue
   ns=[mujoco.mj_id2name(m,mujoco.mjtObj.mjOBJ_BODY,int(m.geom_bodyid[g])) or 'world' for g in c.geom];same=all('left_' in n for n in ns) or all('right_' in n for n in ns)
   if same:continue
   contacts.append({'bodies':ns,'penetration_mm':float(-c.dist*1000)})
  contacts.sort(key=lambda x:-x['penetration_mm']);rows.append({'source_frame':i,'control_frame':frame,'penetrating_cross_body_contacts':contacts,'maximum_penetration_mm':max([x['penetration_mm'] for x in contacts],default=0.)})
  if penetration_epsilon<.001:
   from rigid_pose_collision_audit import PoseCollisionAuditor
   pose=e.state_0.body_q.numpy()
   if pose_auditor is None:pose_auditor=PoseCollisionAuditor(m,d,list(e.model.body_label),pose,cfg['objects'])
   physical=pose_auditor.inspect(pose)
   rows[-1].update(penetrating_cross_body_contacts=[{'bodies':hit['bodies'],'penetration_mm':hit['penetration_m']*1000} for hit in physical['worst_contacts']],maximum_penetration_mm=physical['maximum_penetration_m']*1000,all_required_pairs_checked=True)
 out={'method':__doc__,'frame_count':T,'reference':reference,'maximum_cross_body_penetration_mm':max(x['maximum_penetration_mm'] for x in rows),'source_frames_over_10mm':sum(x['maximum_penetration_mm']>10 for x in rows),'source_frames_over_20mm':sum(x['maximum_penetration_mm']>20 for x in rows),'collision_hull_count':len(hulls),'frames':rows};(R/f'evaluation/{reference}-collision-audit.json').write_text(json.dumps(out,indent=2));print(json.dumps({k:v for k,v in out.items() if k!='frames'},indent=2),flush=True);raise SystemExit(0)
runner.run_training=inspect
sys.argv=[sys.argv[0],'experiment=sharpa_flash_sac',f"task.parquet='/episode/human_motion_data/{reference}/processed/sequence_id={seq}/robot_name=sharpa_wave'",'task.motion_speed=1.0','reset.reset_finger_openness=0.0','scene.world_count=1','logging.mode=disabled','output_dir=/episode/diagnostic-checkpoints','hydra.run.dir=/episode/hydra/reference-collision-audit']
runpy.run_path('/workspace/scripts/train_flash_sac.py',run_name='__main__')
