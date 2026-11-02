"""Diagnostic PD tracking with zero residual actions; this is not a learned policy rollout."""
import os,json,runpy,sys,hydra,mujoco,numpy as np,warp as wp
from pathlib import Path
from omegaconf import OmegaConf
import flash_chord.training.flash_sac.runner as runner
from flash_chord.evaluation.harness import apply_evaluation_composition
R=Path('/episode');c=json.loads((R/'episode-config.json').read_text());reference=os.environ.get('V2D_DIAGNOSTIC_REFERENCE',c.get('training_reference','track3_coherent_variant'));substeps=int(os.environ.get('V2D_DIAGNOSTIC_SUBSTEPS','5'));label=os.environ.get('V2D_DIAGNOSTIC_LABEL','zero-residual-diagnostic'+(f'-substeps-{substeps}' if substeps!=5 else ''))
def inspect(env,*args,**kwargs):
 e=env.env;assert not e.config.auto_reset;e.set_voc_scale(0.);e.reset(frame_id=0);poses=[];frames=[];assist=[];action=wp.zeros_like(e.action_input);contact_rows=[];probe=mujoco.MjData(e.solver.mj_model);initial=e.state_0.body_q.numpy().copy()
 for i in range(e.reference.num_frames):
  frame=int(e.timestep.numpy()[0]);assert frame==i,(frame,i);e.step(action);poses.append(e.state_0.body_q.numpy().copy());frames.append(frame);assist.append(float(e.voc_scale.numpy()[0]))
  if i<e.reference.num_frames:
   e.solver._update_mjc_data(probe,e.model,e.state_0);mujoco.mj_forward(e.solver.mj_model,probe);hits=[]
   for contact in probe.contact:
    ns=[mujoco.mj_id2name(e.solver.mj_model,mujoco.mjtObj.mjOBJ_BODY,int(e.solver.mj_model.geom_bodyid[j])) or 'world' for j in contact.geom]
    if any(any(name in n for name in c['objects']) for n in ns):hits.append({'bodies':ns,'distance_m':float(contact.dist)})
   contact_rows.append({'control_frame':i,'contacts':hits})
 (R/f'evaluation/{label}-contacts.json').write_text(json.dumps(contact_rows,indent=2))
 assert np.all(np.array(assist)==0.)
 np.savez_compressed(R/f'evaluation/{label}-states.npz',body_poses=poses,initial_body_poses=initial,body_names=e.model.body_label,reference_frames=frames,object_assistance=assist,trajectory_kind=__doc__)
 from scipy.spatial.transform import Rotation
 rest=[];names=list(e.model.body_label);trajectory=np.array(poses)
 for b,name in enumerate(c['objects']):
  start=min([g['start'] for g in c['grasps'] if g['object']==b],default=c['frames']);cutoff=min(len(poses),max(0,2*(start-3)))
  if cutoff==0:continue
  touch=next((x['control_frame'] for x in contact_rows if any(hit['distance_m']<0 and any(name in n for n in hit['bodies']) and any('sharpa_wave' in n for n in hit['bodies']) for hit in x['contacts'])),None);cutoff=min(cutoff,touch) if touch is not None else cutoff
  if cutoff==0:continue
  k=names.index(f'track3_{name}/object');position=np.linalg.norm(trajectory[:cutoff,k,:3]-initial[k,:3],axis=1);angle=(Rotation.from_quat(initial[k,3:]).inv()*Rotation.from_quat(trajectory[:cutoff,k,3:])).magnitude();bad=(position>.01)|(angle>.2)
  rest.append({'object':name,'control_frames_before_approach':cutoff,'first_hand_contact_control_frame':touch,'maximum_position_drift_m':float(position.max()),'maximum_orientation_drift_rad':float(angle.max()),'first_unstable_control_frame':int(np.flatnonzero(bad)[0]) if bad.any() else None,'passed_1cm_0_2rad_passive_rest':bool(not bad.any())})
 (R/f'evaluation/{label}-initial-rest.json').write_text(json.dumps({'diagnostic_only':True,'policy_inference':False,'reference':reference,'objects':rest},indent=2))
 (R/f'evaluation/{label}-protocol.json').write_text(json.dumps({'kind':__doc__,'policy_inference':False,'residual_actions':0.,'object_assistance':0.,'failure_resets':False,'reference':reference,'frames':len(frames),'all_states_finite':bool(np.isfinite(poses).all())},indent=2));print(label,'finite',bool(np.isfinite(poses).all()),'initial rests',rest,flush=True);raise SystemExit(0)
runner.run_training=inspect
sys.argv=[sys.argv[0],'experiment=sharpa_flash_sac',f"task.parquet='/episode/human_motion_data/{reference}/processed/sequence_id={c['sequence']}/robot_name=sharpa_wave'",'task.motion_speed=0.5',f'sim.substeps={substeps}','reset.reset_finger_openness=0.0','scene.world_count=1','sim.njmax=1536','logging.mode=disabled','output_dir=/episode/diagnostic-checkpoints','hydra.run.dir=/episode/hydra/zero-residual-diagnostic']
native=runpy.run_path('/workspace/scripts/train_flash_sac.py',run_name='native_training')
@hydra.main(version_base='1.3',config_path='/workspace/src/flash_chord/configs',config_name='train_flash_sac')
def main(cfg):
 cfg=OmegaConf.create(apply_evaluation_composition(OmegaConf.to_container(cfg,resolve=True),world_count=1,reset_mode='explicit'));native['main'].__wrapped__(cfg)
main()
