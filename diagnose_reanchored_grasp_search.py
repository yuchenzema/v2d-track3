"""Reanchor a physically settled grasp to the observed object pose, then retest it."""
import json,os,runpy,itertools
from pathlib import Path
import numpy as np
from scipy.spatial.transform import Rotation
import warp as wp
from flash_chord.envs.base import BaseEnv
import flash_chord.training.flash_sac.runner as runner

ROOT=Path('/episode');cfg=json.loads((ROOT/'episode-config.json').read_text())
source=int(os.environ.get('V2D_STATIC_GRASP_SOURCE_FRAME','65'))
label=f'refined-grasp-search-source-{source}'
previous=json.loads((ROOT/f'evaluation/{label}.json').read_text())
data=np.load(ROOT/f'evaluation/{label}.npz')
W=128;os.environ['V2D_STATIC_GRASP_WORLD_COUNT']=str(W)
os.environ['V2D_STATIC_GRASP_SOURCE_FRAME']=str(source)
os.environ['V2D_WRIST_MOTOR_PROFILE']=json.dumps({'kp':250.,'kd':20.})
native=runpy.run_path('/episode/scripts/diagnose_static_grasp.py',run_name='static_probe_library')

def matrix(pose):
    m=np.eye(4);m[:3,:3]=Rotation.from_quat(pose[3:]).as_matrix();m[:3,3]=pose[:3];return m

def probe(env,*args,**kwargs):
    e=env.env;B=e.model.body_count//W;Q=e.model.joint_coord_count//W
    frame=round(source*(e.reference.num_frames-1)/(cfg['frames']-1))
    assert not e.config.auto_reset and e.world_count==W
    layout=e.action.buffers.layout;scale=e.action.buffers.scale.numpy()
    object_ids=e.command.body_ids_w.numpy().reshape(W,len(cfg['objects']))
    results={};saved={'body_names':list(e.model.body_label[:B])}
    for name,best in previous['best_candidates'].items():
        body=cfg['objects'].index(name);side=best['side'];hand=next(h for h in e.scene.layout.hands if h.side==side)
        start=layout.hand_action_starts[layout.sides.index(side)]
        names=e.reference.finger_joint_names(side);finger=np.asarray(e.reference.finger_joint_pos(side)[frame])[[names.index(n) for n in hand.finger_joint_names]]
        wrist=np.asarray(e.reference.wrist_pos_w(side)[frame]);rotation=Rotation.from_quat(e.reference.wrist_quat_w(side)[frame],scalar_first=True)
        O=np.eye(4);O[:3,:3]=Rotation.from_quat(e.reference.object_body_quat_w()[frame,body],scalar_first=True).as_matrix();O[:3,3]=e.reference.object_body_pos_w()[frame,body]
        local=np.asarray(best['wrist_in_object']);desired=O@local
        old_states=data[name+'_body_poses'];old_names=data['body_names'].tolist();oi=old_names.index(f'track3_{name}/object')
        proposals=[None]+list(itertools.product([19,59],[.25,.5,.75,1.],[-.005,0.,.005],[-.05,0.,.05]))
        candidates=[];raw=np.zeros((W,layout.action_dim),np.float32);offsets=np.zeros((W,Q),np.float32)
        low=e.model.joint_limit_lower.numpy()[list(hand.finger_dof_ids)];high=e.model.joint_limit_upper.numpy()[list(hand.finger_dof_ids)]
        for world in range(W):
            proposal=proposals[world] if world<len(proposals) else None
            target=desired.copy();f=np.asarray(best['finger_joints']).copy()
            if proposal is not None:
                step,alpha,lift_bias,closing=proposal
                goal=desired.copy();goal[2,3]+=.03 if step==59 else 0.
                revised=O@np.linalg.inv(matrix(old_states[step,oi]))@goal
                target[:3,3]=(1-alpha)*desired[:3,3]+alpha*revised[:3,3]
                delta=Rotation.from_matrix(desired[:3,:3]).inv()*Rotation.from_matrix(revised[:3,:3])
                target[:3,:3]=(Rotation.from_matrix(desired[:3,:3])*Rotation.from_rotvec(alpha*delta.as_rotvec())).as_matrix()
                target[2,3]+=lift_bias
                for j,n in enumerate(hand.finger_joint_names):
                    if n.endswith(('_MCP_FE','_PIP','_DIP','_IP')):f[j]+=closing
            f=np.clip(f,low+1e-6,high-1e-6)
            delta_pos=target[:3,3]-wrist;delta_rot=(rotation.inv()*Rotation.from_matrix(target[:3,:3])).as_rotvec()
            raw[world,start:start+3]=delta_pos/scale[start:start+3]
            raw[world,start+3:start+6]=delta_rot/scale[start+3:start+6]
            raw[world,start+6:start+6+len(f)]=(f-finger)/scale[start+6:start+6+len(f)]
            offsets[world,list(hand.wrist_pos_q_ids)]=delta_pos;q=hand.wrist_orient_q_id
            offsets[world,q:q+4]=Rotation.from_matrix(target[:3,:3]).as_quat()-rotation.as_quat()
            offsets[world,list(hand.finger_q_ids)]=f-finger
            candidates.append((np.linalg.inv(O)@target,f))
        e.set_voc_scale(0.);e.reset(frame_id=frame);e.reference_joint_q_offset.assign(offsets.reshape(-1));BaseEnv.reset(e,frame_id=frame)
        if os.environ.get('V2D_GRASP_CONSISTENT_INITIAL_MOTORS')=='1':e.action.buffers.filtered_action.assign(raw.reshape(-1))
        initial=e.state_0.body_q.numpy().reshape(W,B,7).copy()
        assert np.allclose(initial[np.arange(W),object_ids[:,body]%B,:3],O[:3,3],atol=1e-5)
        states=[]
        for step in range(60):
            e.timestep.fill_(frame);action=raw.copy();action[:,start+2]+=.03*float(np.clip((step-19)/20.,0.,1.))/scale[start+2]
            e.step(wp.array(action.reshape(-1),device=e.device));assert np.all(e.voc_scale.numpy()==0.)
            states.append(e.state_0.body_q.numpy().reshape(W,B,7).copy())
        states=np.asarray(states);assert np.isfinite(states).all()
        oi=object_ids[:,body]%B;wi=hand.palm_frame.body_id
        obj=states[:,np.arange(W),oi];p0=initial[np.arange(W),oi]
        drift=np.linalg.norm(obj[19,:,:3]-p0[:,:3],axis=1);hand_lift=states[-1,:,wi,:3]-states[19,:,wi,:3]
        object_lift=obj[-1,:,:3]-obj[19,:,:3];follow=np.linalg.norm(hand_lift-object_lift,axis=1)
        angle=(Rotation.from_quat(p0[:,3:]).inv()*Rotation.from_quat(obj[-1,:,3:])).magnitude()
        stable=(drift<=.015)&(follow<=.01)&(angle<=.2)&(hand_lift[:,2]>=.02)&(object_lift[:,2]>=.015)
        cost=drift+follow+.04*angle+2*np.maximum(0.,.02-hand_lift[:,2])+2*np.maximum(0.,.015-object_lift[:,2])
        rows=[]
        for j,proposal in enumerate(proposals):
            rows.append({'candidate':j,'parameters':proposal,'stable_hold_and_lift':bool(stable[j]),
                'static_position_drift_m':float(drift[j]),'object_following_error_m':float(follow[j]),
                'object_orientation_change_rad':float(angle[j]),'actual_hand_lift_m':float(hand_lift[j,2]),
                'actual_object_lift_m':float(object_lift[j,2]),'cost':float(cost[j])})
        choice=min(range(len(proposals)),key=lambda j:(not stable[j],cost[j]));local,fingers=candidates[choice]
        results[name]={**rows[choice],'side':side,'object_index':body,'source_frame':source,
            'finger_joint_names':list(hand.finger_joint_names),'finger_joints':fingers.tolist(),'wrist_in_object':local.tolist(),
            'trials':rows}
        saved[name+'_body_poses']=states[:,choice];saved[name+'_initial_body_poses']=initial[choice]
        print(name,'reanchored', {k:v for k,v in results[name].items() if k not in ['trials','finger_joints','finger_joint_names','wrist_in_object']},flush=True)
    report={'episode':cfg['episode'],'diagnostic_only':True,'policy_inference':False,'object_assistance':0.,
        'source_object_poses_unchanged':True,'robot_pose_adjustments_only':True,'resets_during_probe':False,'best_candidates':results}
    (ROOT/'evaluation/reanchored-grasp-search.json').write_text(json.dumps(report,indent=2))
    np.savez_compressed(ROOT/'evaluation/reanchored-grasp-search.npz',**saved)
    raise SystemExit(0)

runner.run_training=probe;native['native']['main'].__wrapped__.__globals__['run_training']=probe;native['main']()
