"""Search physically valid initial hand postures; this is not a policy rollout."""
import itertools
import json
import os
from pathlib import Path
import runpy
import numpy as np
from scipy.spatial.transform import Rotation
import trimesh
import warp as wp
from flash_chord.envs.base import BaseEnv
import flash_chord.training.flash_sac.runner as runner

ROOT=Path('/episode');cfg=json.loads((ROOT/'episode-config.json').read_text())
WORLD_COUNT=128
os.environ['V2D_STATIC_GRASP_WORLD_COUNT']=str(WORLD_COUNT)
os.environ['V2D_WRIST_MOTOR_PROFILE']=json.dumps({'kp':250.,'kd':20.})
native=runpy.run_path('/episode/scripts/diagnose_static_grasp.py',run_name='static_probe_library')


def probe(env,*args,**kwargs):
    e=env.env;W=e.world_count;B=e.model.body_count//W;Q=e.model.joint_coord_count//W
    assert W==WORLD_COUNT and not e.config.auto_reset
    frame=int(round(native['source_frame']*(e.reference.num_frames-1)/(cfg['frames']-1)))
    object_ids=e.command.body_ids_w.numpy().reshape(W,len(cfg['objects']))
    names=list(e.model.body_label[:B]);layout=e.action.buffers.layout;scales=e.action.buffers.scale.numpy()
    results={};best_states={};best_initial={};best_parameters={}
    for body,name in enumerate(cfg['objects']):
        if os.environ.get('V2D_GRASP_SEARCH_OBJECTS') and name not in json.loads(os.environ['V2D_GRASP_SEARCH_OBJECTS']):continue
        side=next(g['side'] for g in cfg['grasps'] if g['object']==body and g['start']<=native['source_frame']<=g['end'])
        hand=next(h for h in e.scene.layout.hands if h.side==side)
        start=layout.hand_action_starts[layout.sides.index(side)]
        ref_names=e.reference.finger_joint_names(side);ref_index={n:i for i,n in enumerate(ref_names)}
        original_f=np.asarray(e.reference.finger_joint_pos(side)[frame])[[ref_index[n] for n in hand.finger_joint_names]]
        wrist=np.asarray(e.reference.wrist_pos_w(side)[frame]);wrist_R=Rotation.from_quat(e.reference.wrist_quat_w(side)[frame],scalar_first=True).as_matrix()
        center=trimesh.load(ROOT/f'reconstruction/mesh_priors/{name}.ply',process=False).vertices.mean(0)
        object_R=Rotation.from_quat(e.reference.object_body_quat_w()[frame,body],scalar_first=True).as_matrix()
        direction=object_R@center+e.reference.object_body_pos_w()[frame,body]-wrist;direction/=np.linalg.norm(direction)
        if name in ['mini_sweeper','mini_dust_pan','plastic_dish_rack']:
            proposals=[None]+list(itertools.product([0.,.015,.03],[.4,.8,1.2],[1.,1.3,1.6],[.8,1.1,1.35],[.3,.6,.9],[.8,1.2,1.6]))
        else:
            proposals=[None]+list(itertools.product([0.,.015,.03],[.2,.5,.8],[.6,.9,1.2],[.3,.7,1.1],[0.,.3,.6],[.3,.8,1.3]))
        rows=[];best=None
        for beginning in range(0,len(proposals),W):
            batch=proposals[beginning:beginning+W];actual_count=len(batch)
            batch=batch+[batch[-1]]*(W-actual_count)
            offsets=np.zeros((W,Q),np.float32);actions=np.zeros((W,layout.action_dim),np.float32);targets=[]
            for world,proposal in enumerate(batch):
                desired=original_f.copy();distance=0.
                if proposal is not None:
                    distance,mcp,pip,dip,thumb_mcp,thumb_ip=proposal
                    for j,joint in enumerate(hand.finger_joint_names):
                        if 'thumb' in joint:
                            if joint.endswith('_MCP_FE'):desired[j]=thumb_mcp
                            elif joint.endswith('_IP'):desired[j]=thumb_ip
                        elif joint.endswith('_MCP_FE'):desired[j]=mcp
                        elif joint.endswith('_PIP'):desired[j]=pip
                        elif joint.endswith('_DIP'):desired[j]=dip
                delta=desired-original_f
                offsets[world,list(hand.finger_q_ids)]=delta
                offsets[world,list(hand.wrist_pos_q_ids)]=direction*distance
                actions[world,start:start+3]=direction*distance/scales[start:start+3]
                actions[world,start+6:start+6+len(desired)]=delta/scales[start+6:start+6+len(desired)]
                targets.append(desired)
            allowed=set(hand.finger_q_ids)|set(hand.wrist_pos_q_ids)
            assert not offsets[:,[q for q in range(Q) if q not in allowed]].any()
            e.set_voc_scale(0.);e.reset(frame_id=frame)
            e.reference_joint_q_offset.assign(offsets.reshape(-1))
            BaseEnv.reset(e,frame_id=frame)
            initial=e.state_0.body_q.numpy().copy()
            expected=np.asarray(e.reference.object_body_pos_w())[frame]
            assert np.allclose(initial[object_ids,:3],expected[None,:,:],atol=1e-5)
            states=[]
            for step in range(60):
                e.timestep.fill_(frame);lift=.03*float(np.clip((step-19)/20.,0.,1.));raw=actions.copy()
                raw[:,start+2]+=lift/scales[start+2]
                e.step(wp.array(raw.reshape(-1),device=e.device))
                assert np.all(e.voc_scale.numpy()==0.)
                states.append(e.state_0.body_q.numpy().copy())
            states=np.asarray(states)
            assert np.isfinite(states).all()
            for world in range(actual_count):
                oi=int(object_ids[world,body]);wi=hand.palm_frame.body_id+world*B
                obj=states[:,oi];p0=initial[oi]
                wrist_lift=states[-1,wi,:3]-states[19,wi,:3];obj_lift=obj[-1,:3]-obj[19,:3]
                drift=float(np.linalg.norm(obj[19,:3]-p0[:3]));follow=float(np.linalg.norm(obj_lift-wrist_lift))
                angle=float((Rotation.from_quat(p0[3:]).inv()*Rotation.from_quat(obj[-1,3:])).magnitude())
                stable=drift<=.015 and follow<=.01 and angle<=.2 and wrist_lift[2]>=.02 and obj_lift[2]>=.015
                proposal=batch[world]
                distance=0. if proposal is None else proposal[0]
                cost=drift+follow+.03*angle+max(0.,.02-float(wrist_lift[2]))+max(0.,.015-float(obj_lift[2]))
                record={'candidate':beginning+world,'parameters':proposal,'stable_hold_and_lift':bool(stable),
                    'static_position_drift_m':drift,'object_following_error_m':follow,
                    'object_orientation_change_rad':angle,'actual_hand_lift_m':float(wrist_lift[2]),
                    'actual_object_lift_m':float(obj_lift[2]),'cost':cost}
                rows.append(record)
                ranking=(not stable,cost+.01*float(np.mean(np.abs(targets[world]-original_f)))+.1*distance)
                if best is None or ranking<best:
                    best=ranking;best_parameters[name]={**record,'side':side,'object_index':body,
                        'finger_joint_names':list(hand.finger_joint_names),'finger_joints':targets[world].tolist(),
                        'world_inward_direction':direction.tolist(),'reference_wrist_position':wrist.tolist(),
                        'reference_wrist_rotation':wrist_R.tolist(),'inward_m':distance,
                        'local_object_body_id':oi-world*B,'local_palm_body_id':wi-world*B}
                    best_states[name]=states[:,world*B:(world+1)*B].copy()
                    best_initial[name]=initial[world*B:(world+1)*B].copy()
            print(name,'tested',min(beginning+W,len(proposals)),'stable',sum(r['stable_hold_and_lift'] for r in rows),
                'best',best_parameters[name],flush=True)
            if best is not None and best[0] is False and best[1]<.012:break
        results[name]=rows
    report={'episode':cfg['episode'],'diagnostic_only':True,'policy_inference':False,
        'source_frame':native['source_frame'],'wrist_motor_profile':{'kp':250.,'kd':20.,'force_limit_N':45.},
        'object_assistance':0.,'no_object_pose_offsets':True,
        'object_state_resets_only_between_independent_probes':True,'steps_per_probe':60,
        'reference_hand_posture_offsets_only_at_initialization':True,
        'best_candidates':best_parameters,'trials':results,'approved_for_training':False}
    (ROOT/'evaluation/grasp-pose-search.json').write_text(json.dumps(report,indent=2))
    saved={'body_names':names}
    for name in best_states:saved[name+'_body_poses']=best_states[name];saved[name+'_initial_body_poses']=best_initial[name]
    np.savez_compressed(ROOT/'evaluation/grasp-pose-search.npz',**saved)
    raise SystemExit(0)


runner.run_training=probe
native['native']['main'].__wrapped__.__globals__['run_training']=probe
native['main']()
