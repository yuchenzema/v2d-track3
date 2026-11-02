"""Find brush-plus-right-hand poses clear of the native rack and table geometry."""
import itertools,json,runpy,sys
from pathlib import Path
import mujoco,numpy as np,trimesh
from scipy.spatial.transform import Rotation
import flash_chord.training.flash_sac.runner as runner
R=Path('/episode');cfg=json.loads((R/'episode-config.json').read_text());REF='track3_consistent_brush_pickup_candidate'
def inspect(env,*args,**kwargs):
 e=env.env;s=e.solver;m=s.mj_model;d=s.mj_data;frame=89;e.reset(frame_id=frame);s._update_mjc_data(d,e.model,e.state_0);mujoco.mj_forward(m,d);q0=d.qpos.copy()
 body=mujoco.mj_name2id(m,mujoco.mjtObj.mjOBJ_BODY,'track3_mini_sweeper_object');adr=int(m.jnt_qposadr[int(m.body_jntadr[body])]);p0=q0[adr:adr+3];q=Rotation.from_quat(q0[adr+3:adr+7],scalar_first=True)
 def joint(name):return int(m.jnt_qposadr[mujoco.mj_name2id(m,mujoco.mjtObj.mjOBJ_JOINT,name)])
 pos=[joint('right_wrist_'+axis) for axis in 'xyz'];ori=joint('right_wrist_rotvec');w0=q0[pos];wq=Rotation.from_quat(q0[ori:ori+4],scalar_first=True)
 np.testing.assert_allclose(w0,e.reference.wrist_pos_w('right')[frame],atol=1e-6)
 centre=np.asarray(trimesh.load(R/'reconstruction/mesh_priors/mini_sweeper.ply',process=False).vertices).mean(0);c0=q.apply(centre)+p0
 assembly={body}|{i for i in range(m.nbody) if (mujoco.mj_id2name(m,mujoco.mjtObj.mjOBJ_BODY,i) or '').startswith('right_sharpa_wave_')}
 def penetration():
  maximum=0.
  for contact in d.contact:
   a,b=[int(m.geom_bodyid[g]) for g in contact.geom]
   if (a in assembly)==(b in assembly):continue
   maximum=max(maximum,-float(contact.dist))
  return maximum
 feasible=[];closest=None;count=0
 for angles in itertools.product([-.3,0.,.3],[-.3,0.,.3],[-.6,-.3,0.,.3,.6]):
  rotation=q*Rotation.from_rotvec(angles);dr=(rotation*q.inv()).as_matrix()
  for shift in itertools.product([-.03,0.,.03],[-.03,0.,.03],[0.,.02,.04,.06]):
   p=c0+shift-rotation.apply(centre);dt=p-dr@p0;d.qpos[:]=q0;d.qpos[adr:adr+3]=p;d.qpos[adr+3:adr+7]=rotation.as_quat(scalar_first=True);d.qpos[pos]=dr@w0+dt;d.qpos[ori:ori+4]=(Rotation.from_matrix(dr)*wq).as_quat(scalar_first=True);mujoco.mj_forward(m,d)
   depth=penetration();row={'object_position':p.tolist(),'object_wxyz':rotation.as_quat(scalar_first=True).tolist(),'centroid_shift_m':list(shift),'local_rotation_rad':list(angles),'maximum_assembly_penetration_m':depth};count+=1
   if closest is None or depth<closest['maximum_assembly_penetration_m']:closest=row
   if depth<=.0015:feasible.append(row)
 report={'diagnostic_only':True,'source_frame':frame,'candidate_count':count,'collision_feasible_count':len(feasible),'minimum_penetration_candidate':closest,'maximum_allowed_assembly_penetration_m':.0015,'assembly_members':'Brush and all right-hand collision bodies; intended hand/brush contacts excluded; rack, table and left hand checked.','feasible_candidates':feasible}
 (R/'research/brush-assembly-clearance-candidates.json').write_text(json.dumps(report,indent=2));print('Assembly candidates',count,'feasible',len(feasible),'minimum penetration',closest['maximum_assembly_penetration_m'],flush=True);raise SystemExit(0)
runner.run_training=inspect
sys.argv=[sys.argv[0],'experiment=sharpa_flash_sac',f"task.parquet='/episode/human_motion_data/{REF}/processed/sequence_id={cfg['sequence']}/robot_name=sharpa_wave'",'task.motion_speed=1.0','reset.reset_finger_openness=0.0','scene.world_count=1','logging.mode=disabled','output_dir=/episode/diagnostic-checkpoints','hydra.run.dir=/episode/hydra/brush-assembly-clearance']
runpy.run_path('/workspace/scripts/train_flash_sac.py',run_name='__main__')
