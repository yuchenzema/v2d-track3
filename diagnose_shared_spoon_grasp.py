"""Test a continuous spoon grip before insertion and inside the holder.

Diagnostic only:16 independent candidates, two source poses, ordinary robot
motors and zero object assistance. Objects are reset only between probes.
"""
import os,json,runpy,numpy as np
from pathlib import Path
from scipy.spatial.transform import Rotation,Slerp
import warp as wp
from flash_chord.envs.base import BaseEnv
import flash_chord.training.flash_sac.runner as runner
r=Path('/episode');cfg=json.loads((r/'episode-config.json').read_text());W=16
os.environ.update(V2D_STATIC_GRASP_WORLD_COUNT=str(W),V2D_STATIC_GRASP_SOURCE_FRAME='74',V2D_GRASP_PROBE_REFERENCE=cfg['training_reference'],
 V2D_WRIST_MOTOR_PROFILE='{"kp":25,"kd":6}',V2D_WRIST_SIDE_MOTOR_PROFILES='{"left":{"kp":25,"kd":6}}',V2D_STATIC_GRASP_LIFT_SIDES='["right"]')
native=runpy.run_path('/episode/scripts/diagnose_static_grasp.py',run_name='static_probe_library')
def probe(environment,*args,**kwargs):
 e=environment.env;B=e.model.body_count//W;Q=e.model.joint_coord_count//W;hand=next(h for h in e.scene.layout.hands if h.side=='right');b=cfg['objects'].index('wooden_spoon')
 best=json.loads((r/'evaluation/continuous-spoon-low-impedance-search.json').read_text())['best_candidates']['wooden_spoon'];assert best['stable_hold_and_lift']
 base=np.array(best['wrist_in_object']);f0=np.array(best['finger_joints']);slides=np.repeat([0.,.02,.04,.06],4);extra=np.tile([0.,.04,-.04,.08],4);weights=np.zeros(W)
 local=np.tile(base,(W,1,1));local[:,2,3]+=slides
 flex=np.array([n.endswith(('_FE','_IP','_PIP','_DIP')) for n in hand.finger_joint_names]);fingers=f0[None,:]+extra[:,None]*flex
 lo=e.model.joint_limit_lower.numpy()[list(hand.finger_dof_ids)];hi=e.model.joint_limit_upper.numpy()[list(hand.finger_dof_ids)];fingers=np.clip(fingers,lo+1e-6,hi-1e-6)
 layout=e.action.buffers.layout;scale=e.action.buffers.scale.numpy();start=layout.hand_action_starts[layout.sides.index('right')];objects=e.command.body_ids_w.numpy().reshape(W,len(cfg['objects']));results=[]
 for t in [74,103]:
  frame=round(t*(e.reference.num_frames-1)/(cfg['frames']-1));e.reference_joint_q_offset.zero_();e.set_voc_scale(0.);e.reset(frame_id=frame)
  O=np.eye(4);O[:3,3]=e.reference.object_body_pos_w()[frame,b];O[:3,:3]=Rotation.from_quat(e.reference.object_body_quat_w()[frame,b],scalar_first=True).as_matrix();world=O@local
  refp=np.asarray(e.reference.wrist_pos_w('right')[frame]);refr=Rotation.from_quat(e.reference.wrist_quat_w('right')[frame],scalar_first=True);names=e.reference.finger_joint_names('right');reff=np.asarray(e.reference.finger_joint_pos('right')[frame])[[names.index(n) for n in hand.finger_joint_names]]
  delta=world[:,:3,3]-refp;rv=(refr.inv()*Rotation.from_matrix(world[:,:3,:3])).as_rotvec();offsets=np.zeros((W,Q),np.float32);raw=np.zeros((W,layout.action_dim),np.float32)
  offsets[:,list(hand.wrist_pos_q_ids)]=delta;q=hand.wrist_orient_q_id;offsets[:,q:q+4]=Rotation.from_matrix(world[:,:3,:3]).as_quat()-refr.as_quat();offsets[:,list(hand.finger_q_ids)]=fingers-reff
  raw[:,start:start+3]=delta/scale[start:start+3];raw[:,start+3:start+6]=rv/scale[start+3:start+6];raw[:,start+6:start+6+len(reff)]=(fingers-reff)/scale[start+6:start+6+len(reff)]
  e.reference_joint_q_offset.assign(offsets.ravel());BaseEnv.reset(e,frame_id=frame)
  # Start the diagnostic controller at the same grip as its initial robot pose.
  # Otherwise EMA pulls a20cm-offset probe toward the unrelated original grip.
  e.action.buffers.filtered_action.assign(raw.ravel())
  initial=e.state_0.body_q.numpy().reshape(W,B,7).copy();states=[]
  for j in range(len(cfg['objects'])):assert np.allclose(initial[np.arange(W),objects[:,j]%B,:3],e.reference.object_body_pos_w()[frame,j],atol=1e-5)
  for step in range(60):
   e.timestep.fill_(frame);action=raw.copy();action[:,start+2]+=.03*float(np.clip((step-19)/20.,0,1))/scale[start+2];e.step(wp.array(action.ravel(),device=e.device));assert np.all(e.object_control_scale.numpy()==0.);states.append(e.state_0.body_q.numpy().reshape(W,B,7).copy())
  states=np.array(states);oi=objects[:,b]%B;wi=hand.palm_frame.body_id;obj=states[:,np.arange(W),oi];p0=initial[np.arange(W),oi]
  drift=np.linalg.norm(obj[19,:,:3]-p0[:,:3],axis=1);lift=states[-1,:,wi,:3]-states[19,:,wi,:3];olift=obj[-1,:,:3]-obj[19,:,:3];follow=np.linalg.norm(lift-olift,axis=1);angle=(Rotation.from_quat(p0[:,3:]).inv()*Rotation.from_quat(obj[-1,:,3:])).magnitude()
  rack=cfg['objects'].index('plastic_dish_rack');ri=objects[:,rack]%B;rp=initial[np.arange(W),ri];rack_drift=np.linalg.norm(states[-1,np.arange(W),ri,:3]-rp[:,:3],axis=1);rack_angle=(Rotation.from_quat(rp[:,3:]).inv()*Rotation.from_quat(states[-1,np.arange(W),ri,3:])).magnitude()
  stable=(drift<=.015)&(follow<=.01)&(angle<=.2)&(lift[:,2]>=.02)&(olift[:,2]>=.015)&(rack_drift<=.008)&(rack_angle<=.12)
  rows=[{'source_frame':t,'candidate':j,'stable_hold_and_lift':bool(stable[j]),'static_position_drift_m':float(drift[j]),'object_following_error_m':float(follow[j]),'orientation_change_rad':float(angle[j]),'hand_lift_m':float(lift[j,2]),'object_lift_m':float(olift[j,2]),'rack_drift_m':float(rack_drift[j]),'rack_orientation_change_rad':float(rack_angle[j])} for j in range(W)];results.append(rows);print(t,'stable',np.flatnonzero(stable).tolist(),flush=True)
 candidates=[]
 for j in range(W):
  rows=[v[j] for v in results];good=all(v['stable_hold_and_lift'] for v in rows);cost=max(v['static_position_drift_m']+v['object_following_error_m']+.04*v['orientation_change_rad'] for v in rows)
  candidates.append({'candidate':j,'shared_grasp_feasible_at_both_poses':good,'cost':cost,'axial_slide_from_supported_grip_m':float(slides[j]),'finger_flexion_delta_rad':float(extra[j]),'wrist_in_object':local[j].tolist(),'finger_joints':fingers[j].tolist(),'finger_joint_names':list(hand.finger_joint_names),'probes':rows})
 feasible=[x for x in candidates if x['shared_grasp_feasible_at_both_poses']];best=min(feasible,key=lambda x:x['cost']) if feasible else None
 report={'method':__doc__,'diagnostic_only':True,'policy_inference':False,'object_assistance':0.,'robot_controller_initialized_at_probe_grip':True,'object_resets_during_each_probe':False,'source_frames':[74,103],'best':best,'candidates':candidates}
 (r/'research/shared-spoon-two-pose-probe.json').write_text(json.dumps(report,indent=2));print('Shared feasible grasp',None if best is None else {k:best[k] for k in ['candidate','cost','axial_slide_from_supported_grip_m','finger_flexion_delta_rad']},flush=True);raise SystemExit(0)
runner.run_training=probe
native['native']['main'].__wrapped__.__globals__['run_training']=probe
if __name__=='__main__':native['main']()
