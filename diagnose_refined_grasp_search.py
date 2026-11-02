"""Search wrist pose and individual finger joints using physical hold/lift tests."""
import json
import os
from pathlib import Path
import runpy
import numpy as np
from scipy.spatial.transform import Rotation
import warp as wp
from flash_chord.envs.base import BaseEnv
import flash_chord.training.flash_sac.runner as runner
from grasp_search_protocol import search_seed

ROOT=Path('/episode');cfg=json.loads((ROOT/'episode-config.json').read_text())
LABEL=os.environ.get('V2D_REFINED_GRASP_LABEL','refined-grasp-search')
assert LABEL.replace('-','').isalnum()
previous=json.loads((ROOT/'evaluation/grasp-pose-search.json').read_text())
W=128
os.environ['V2D_STATIC_GRASP_WORLD_COUNT']=str(W)
os.environ.setdefault('V2D_WRIST_MOTOR_PROFILE',json.dumps({'kp':250.,'kd':20.}))
if cfg.get('wrist_side_motor_profiles'):
    os.environ['V2D_WRIST_SIDE_MOTOR_PROFILES']=json.dumps(cfg['wrist_side_motor_profiles'])
if os.environ.get('V2D_GRASP_WRIST_SIDE_PROFILES'):
    os.environ['V2D_WRIST_SIDE_MOTOR_PROFILES']=os.environ['V2D_GRASP_WRIST_SIDE_PROFILES']
native=runpy.run_path('/episode/scripts/diagnose_static_grasp.py',run_name='static_probe_library')


def probe(env,*args,**kwargs):
    e=env.env;B=e.model.body_count//W;Q=e.model.joint_coord_count//W
    if os.environ.get('V2D_GRASP_ANGULAR_SIDE_KP'):
        angular=json.loads(os.environ['V2D_GRASP_ANGULAR_SIDE_KP'])
        assert set(angular)=={'left','right'} and all(25<=v<=250 for v in angular.values())
        e.action.wrist.kp.assign(np.asarray([angular[h.side] for h in e.scene.layout.hands],np.float32))
        assert np.all(e.action.wrist.tau_max.numpy()==45.)
    assert e.world_count==W and not e.config.auto_reset
    frame=round(native['source_frame']*(e.reference.num_frames-1)/(cfg['frames']-1))
    objects=e.command.body_ids_w.numpy().reshape(W,len(cfg['objects']))
    layout=e.action.buffers.layout;scale=e.action.buffers.scale.numpy();clips=e.action.buffers.clip.numpy()
    collision_auditor=None
    require_clearance=bool(cfg.get('grasp_search_require_nonpenetrating_initial_pose'))
    random_seed=search_seed(cfg['episode'],LABEL)
    rng=np.random.default_rng(random_seed);results={};saved={'body_names':list(e.model.body_label[:B])}
    for body,name in enumerate(cfg['objects']):
        if name not in previous['best_candidates']:continue
        if os.environ.get('V2D_GRASP_SEARCH_OBJECTS') and name not in json.loads(os.environ['V2D_GRASP_SEARCH_OBJECTS']):continue
        if cfg['episode']==30 and name!='mini_sweeper':continue
        if cfg['episode']==20 and name!='water_pitcher' and not os.environ.get('V2D_GRASP_SEARCH_OBJECTS'):continue
        old=previous['best_candidates'][name];side=os.environ.get('V2D_GRASP_SEARCH_SIDE',old['side'])
        assert side in ['left','right']
        hand=next(h for h in e.scene.layout.hands if h.side==side)
        start=layout.hand_action_starts[layout.sides.index(side)]
        refnames=e.reference.finger_joint_names(side)
        finger=np.asarray(e.reference.finger_joint_pos(side)[frame])[[refnames.index(n) for n in hand.finger_joint_names]]
        wrist=np.asarray(e.reference.wrist_pos_w(side)[frame]);rotation=Rotation.from_quat(e.reference.wrist_quat_w(side)[frame],scalar_first=True)
        lower=np.asarray(e.model.joint_limit_lower.numpy())[list(hand.finger_dof_ids)]
        upper=np.asarray(e.model.joint_limit_upper.numpy())[list(hand.finger_dof_ids)]
        translation_bound=float(os.environ.get('V2D_GRASP_TRANSLATION_BOUND_M','.045'))
        rotation_bound=float(os.environ.get('V2D_GRASP_ROTATION_BOUND_RAD','.6'))
        assert .02<=translation_bound<=.25 and .2<=rotation_bound<=3.2
        lo=np.r_[np.full(3,-translation_bound),np.full(3,-rotation_bound),lower+1e-6]
        hi=np.r_[np.full(3,translation_bound),np.full(3,rotation_bound),upper-1e-6]
        action_clip=clips[start:start+6+len(finger)]
        lo=np.maximum(lo,np.r_[-action_clip[:6],finger-action_clip[6:]])
        hi=np.minimum(hi,np.r_[action_clip[:6],finger+action_clip[6:]])
        center=np.r_[np.asarray(old['world_inward_direction'])*old['inward_m'],np.zeros(3),old['finger_joints']]
        if os.environ.get('V2D_GRASP_USE_REFERENCE_SEED')=='1':center=np.r_[np.zeros(6),finger]
        if os.environ.get('V2D_GRASP_SEED_FIT'):
            seed=json.loads((ROOT/os.environ['V2D_GRASP_SEED_FIT']).read_text())['best_candidates'][name]
            assert seed['side']==side
            O=np.eye(4);O[:3,3]=e.reference.object_body_pos_w()[frame,body];O[:3,:3]=Rotation.from_quat(e.reference.object_body_quat_w()[frame,body],scalar_first=True).as_matrix()
            goal=O@np.asarray(seed['wrist_in_object']);goal[2,3]+=float(os.environ.get('V2D_GRASP_SEED_WORLD_LIFT','0.'))
            center=np.r_[goal[:3,3]-wrist,(rotation.inv()*Rotation.from_matrix(goal[:3,:3])).as_euler('xyz'),seed['finger_joints']]
        if os.environ.get('V2D_GRASP_MIN_WRIST_Z_DELTA'):
            minimum=float(os.environ['V2D_GRASP_MIN_WRIST_Z_DELTA']);assert 0.<=minimum<=.035;lo[2]=minimum
        deviation=np.r_[np.full(3,.012),np.full(3,.18),np.full(len(finger),.18)]
        best=None;best_x=None;history=[]
        generations=int(os.environ.get('V2D_GRASP_SEARCH_GENERATIONS','6'));assert 1<=generations<=24
        for generation in range(generations):
            values=np.clip(rng.normal(center,deviation,(W,len(center))),lo,hi)
            values[0]=np.clip(center,lo,hi);values[1]=np.r_[np.zeros(6),finger]
            if best_x is not None:values[2]=best_x
            offsets=np.zeros((W,Q),np.float32);raw=np.zeros((W,layout.action_dim),np.float32)
            offsets[:,list(hand.wrist_pos_q_ids)]=values[:,:3]
            original_xyzw=rotation.as_quat()
            orientations=(rotation*Rotation.from_euler('xyz',values[:,3:6])).as_quat()
            q=hand.wrist_orient_q_id;offsets[:,q:q+4]=orientations-original_xyzw
            offsets[:,list(hand.finger_q_ids)]=values[:,6:]-finger
            raw[:,start:start+6]=values[:,:6]/scale[start:start+6]
            raw[:,start+6:start+6+len(finger)]=(values[:,6:]-finger)/scale[start+6:start+6+len(finger)]
            allowed=set(hand.wrist_pos_q_ids)|set(range(q,q+4))|set(hand.finger_q_ids)
            assert not offsets[:,[j for j in range(Q) if j not in allowed]].any()
            e.set_voc_scale(0.);e.reset(frame_id=frame)
            e.reference_joint_q_offset.assign(offsets.reshape(-1));BaseEnv.reset(e,frame_id=frame)
            if os.environ.get('V2D_GRASP_CONSISTENT_INITIAL_MOTORS')=='1':
                e.action.buffers.filtered_action.assign(np.clip(raw*scale,-clips,clips).reshape(-1).astype(np.float32))
            initial=e.state_0.body_q.numpy().reshape(W,B,7).copy()
            initial_penetration=np.zeros(W)
            if require_clearance:
                import mujoco
                from rigid_pose_collision_audit import PoseCollisionAuditor,NUMERICAL_EPSILON_M
                if collision_auditor is None:
                    model=e.solver.mj_model;data=mujoco.MjData(model)
                    e.solver._update_mjc_data(data,e.model,e.state_0);mujoco.mj_forward(model,data)
                    collision_auditor=PoseCollisionAuditor(model,data,list(e.model.body_label[:B]),initial[0],cfg['objects'])
                initial_penetration=np.array([collision_auditor.inspect(p,focus_group='hand:'+side)['maximum_penetration_m'] for p in initial])
            for j in range(len(cfg['objects'])):
                assert np.allclose(initial[np.arange(W),objects[:,j]%B,:3],e.reference.object_body_pos_w()[frame,j],atol=1e-5)
            states=[]
            for step in range(60):
                e.timestep.fill_(frame);action=raw.copy()
                action[:,start+2]+=.03*float(np.clip((step-19)/20.,0.,1.))/scale[start+2]
                e.step(wp.array(action.reshape(-1),device=e.device))
                assert np.all(e.voc_scale.numpy()==0.)
                states.append(e.state_0.body_q.numpy().reshape(W,B,7).copy())
            states=np.asarray(states);finite=np.isfinite(states).all(axis=(0,2,3))
            if os.environ.get('V2D_GRASP_SEARCH_REJECT_INVALID')!='1':assert finite.all()
            assert finite.any(),'All grasp candidates produced nonfinite physical states'
            oi=objects[:,body]%B;wi=hand.palm_frame.body_id
            obj=states[:,np.arange(W),oi];p0=initial[np.arange(W),oi]
            drift=np.linalg.norm(obj[19,:,:3]-p0[:,:3],axis=1)
            hand_lift=states[-1,:,wi,:3]-states[19,:,wi,:3]
            object_lift=obj[-1,:,:3]-obj[19,:,:3]
            follow=np.linalg.norm(hand_lift-object_lift,axis=1)
            angle=np.full(W,np.inf)
            angle[finite]=(Rotation.from_quat(p0[finite,3:]).inv()*Rotation.from_quat(obj[-1,finite,3:])).magnitude()
            stable=(drift<=.015)&(follow<=.01)&(angle<=.2)&(hand_lift[:,2]>=.02)&(object_lift[:,2]>=.015)
            cost=drift+follow+.04*angle+2*np.maximum(0.,.02-hand_lift[:,2])+2*np.maximum(0.,.015-object_lift[:,2])
            if require_clearance:
                stable&=initial_penetration<=NUMERICAL_EPSILON_M
                cost+=100.*initial_penetration
            other_distance=np.zeros(W);other_angle=np.zeros(W)
            if os.environ.get('V2D_PROTECT_OTHER_OBJECTS')=='1':
                for other in range(len(cfg['objects'])):
                    if other==body:continue
                    bi=objects[:,other]%B;p=initial[np.arange(W),bi]
                    other_distance=np.maximum(other_distance,np.linalg.norm(states[-1,np.arange(W),bi,:3]-p[:,:3],axis=1))
                    other_rot=np.full(W,np.inf)
                    other_rot[finite]=(Rotation.from_quat(p[finite,3:]).inv()*Rotation.from_quat(states[-1,np.arange(W)[finite],bi[finite],3:])).magnitude()
                    other_angle=np.maximum(other_angle,other_rot)
                cost+=2*other_distance+.04*other_angle
                translation_limit=float(os.environ.get('V2D_GRASP_MAX_OTHER_TRANSLATION','.03'))
                rotation_limit=float(os.environ.get('V2D_GRASP_MAX_OTHER_ROTATION','.3'))
                assert .001<=translation_limit<=.03 and .01<=rotation_limit<=.3
                stable&=(other_distance<=translation_limit)&(other_angle<=rotation_limit)
            stable&=finite;cost[~finite]=np.inf
            for j in range(W):
                if not finite[j]:continue
                row={'generation':generation,'world':j,'stable_hold_and_lift':bool(stable[j]),
                    'static_position_drift_m':float(drift[j]),'object_following_error_m':float(follow[j]),
                    'object_orientation_change_rad':float(angle[j]),'actual_hand_lift_m':float(hand_lift[j,2]),
                    'actual_object_lift_m':float(object_lift[j,2]),'cost':float(cost[j])}
                row.update(other_objects_max_translation_m=float(other_distance[j]),other_objects_max_orientation_rad=float(other_angle[j]))
                if require_clearance:row['initial_hand_cross_body_penetration_m']=float(initial_penetration[j])
                rank=(not bool(stable[j]),float(cost[j]))
                if best is None or rank<best:
                    best=rank;best_x=values[j].copy()
                    wrist_transform=np.eye(4);wrist_transform[:3,3]=wrist+values[j,:3]
                    wrist_transform[:3,:3]=(rotation*Rotation.from_euler('xyz',values[j,3:6])).as_matrix()
                    obj_transform=np.eye(4);obj_transform[:3,3]=p0[j,:3];obj_transform[:3,:3]=Rotation.from_quat(p0[j,3:]).as_matrix()
                    results[name]={**row,'side':side,'object_index':body,'source_frame':native['source_frame'],
                        'wrist_in_object':(np.linalg.inv(obj_transform)@wrist_transform).tolist(),
                        'finger_joint_names':list(hand.finger_joint_names),'finger_joints':values[j,6:].tolist(),
                        'wrist_translation_delta_m':values[j,:3].tolist(),'wrist_local_rpy_delta_rad':values[j,3:6].tolist(),
                        'wrist_local_rotvec_delta_rad':Rotation.from_euler('xyz',values[j,3:6]).as_rotvec().tolist()}
                    saved[name+'_body_poses']=states[:,j].copy();saved[name+'_initial_body_poses']=initial[j].copy()
            order=np.lexsort((cost,~stable));elite=values[[j for j in order if finite[j]][:16]]
            center=.3*center+.7*elite.mean(axis=0)
            deviation=np.maximum(.65*deviation+.35*elite.std(axis=0),np.r_[np.full(3,.002),np.full(3,.025),np.full(len(finger),.03)])
            history.append({'generation':generation,'stable_candidates':int(stable.sum()),'rejected_nonfinite_candidates':int((~finite).sum()),'minimum_cost':float(cost.min())})
            print(name,'generation',generation,'stable',int(stable.sum()),'best',results[name],flush=True)
            if not best[0] and best[1]<.012:break
        results[name]['search_history']=history
    report={'episode':cfg['episode'],'diagnostic_only':True,'policy_inference':False,'object_assistance':0.,
        'search_random_seed':random_seed,
        'action_convention':'Native local roll-pitch-yaw residuals; filtered action initialized in scaled physical units',
        'wrist_angular_gains':e.action.wrist.kp.numpy().tolist(),
        'robot_action_filter_initialized_at_probe_grip':os.environ.get('V2D_GRASP_CONSISTENT_INITIAL_MOTORS')=='1',
        'source_reference':native['probe_reference'],'native_world_count':W,
        'object_initial_poses_unchanged':True,'robot_pose_offsets_only_at_probe_initialization':True,
        'other_objects_protected_in_objective':os.environ.get('V2D_PROTECT_OTHER_OBJECTS')=='1',
        'resets_during_probe':False,'wrist_motor_profile':json.loads(os.environ['V2D_WRIST_MOTOR_PROFILE']),'best_candidates':results}
    (ROOT/f'evaluation/{LABEL}.json').write_text(json.dumps(report,indent=2))
    np.savez_compressed(ROOT/f'evaluation/{LABEL}.npz',**saved)
    raise SystemExit(0)


runner.run_training=probe
native['native']['main'].__wrapped__.__globals__['run_training']=probe
native['main']()
