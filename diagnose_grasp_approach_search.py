"""Test complete hand approaches; free objects are never driven or reset within attempts."""
import itertools,json,os,runpy
from pathlib import Path
import numpy as np
from scipy.spatial.transform import Rotation
import warp as wp
from flash_chord.envs.base import BaseEnv
import flash_chord.training.flash_sac.runner as runner

ROOT=Path('/episode');cfg=json.loads((ROOT/'episode-config.json').read_text())
LABEL=os.environ.get('V2D_APPROACH_SEARCH_LABEL','grasp-approach-search')
assert LABEL.replace('-','').isalnum()
fits=json.loads((ROOT/'research/planned-grasp-fits.json').read_text())['grasps']
W=128;os.environ['V2D_STATIC_GRASP_WORLD_COUNT']=str(W)
os.environ['V2D_WRIST_MOTOR_PROFILE']=json.dumps({'kp':250.,'kd':20.})
native=runpy.run_path('/episode/scripts/diagnose_static_grasp.py',run_name='static_probe_library')

def smooth(x):
    x=np.clip(x,0.,1.);return x*x*(3.-2.*x)

def probe(env,*args,**kwargs):
    e=env.env;B=e.model.body_count//W;Q=e.model.joint_coord_count//W;N=e.reference.num_frames
    assert W==e.world_count and not e.config.auto_reset
    layout=e.action.buffers.layout;scale=e.action.buffers.scale.numpy()
    reference_pos=np.asarray(e.reference.object_body_pos_w());reference_quat=np.asarray(e.reference.object_body_quat_w())
    object_ids=e.command.body_ids_w.numpy().reshape(W,len(cfg['objects']))
    object_R=Rotation.from_quat(reference_quat.reshape(-1,4),scalar_first=True).as_matrix().reshape(N,len(cfg['objects']),3,3)
    profiles=[None]+list(itertools.product([-1,4,8,16,32],[2,4,8],[0,4],[0.,.02]))
    assert len(profiles)<=W
    raw=np.zeros((N,W,layout.action_dim),np.float32);offsets=np.zeros((W,Q),np.float32)
    source=np.arange(N)*(cfg['frames']-1)/(N-1)
    for fit in fits:
        side=fit['side'];body=fit['object_index'];start,end=fit['held_interval']
        hand=next(h for h in e.scene.layout.hands if h.side==side)
        a=layout.hand_action_starts[layout.sides.index(side)]
        names=e.reference.finger_joint_names(side);idx=[names.index(n) for n in hand.finger_joint_names]
        fingers=np.asarray(e.reference.finger_joint_pos(side))[:,idx]
        positions=np.asarray(e.reference.wrist_pos_w(side));rotations=Rotation.from_quat(e.reference.wrist_quat_w(side),scalar_first=True)
        local=np.asarray(fit['wrist_in_object']);desired_fingers=np.asarray(fit['finger_joints'])
        desired_positions=np.einsum('tij,j->ti',object_R[:,body],local[:3,3])+reference_pos[:,body]
        desired_rotations=Rotation.from_matrix(object_R[:,body]@local[:3,:3])
        first=round(start*(N-1)/(cfg['frames']-1));last=round(end*(N-1)/(cfg['frames']-1))
        pre_delta=desired_rotations[first]*rotations[first].inv()
        post_delta=desired_rotations[last]*rotations[last].inv()
        target_positions=desired_positions.copy();target_rotations=desired_rotations.as_quat()
        before=source<start;after=source>end
        target_positions[before]=desired_positions[first]+pre_delta.apply(positions[before]-positions[first])
        target_rotations[before]=(pre_delta*rotations[before]).as_quat()
        target_positions[after]=desired_positions[last]+post_delta.apply(positions[after]-positions[last])
        target_rotations[after]=(post_delta*rotations[after]).as_quat()
        local_rotation_delta=(rotations.inv()*Rotation.from_quat(target_rotations)).as_rotvec()
        for world,profile in enumerate(profiles):
            if profile is None:continue
            width,closing,delay,arc=profile
            wrist_blend=np.ones(N) if width==-1 else smooth((source-start+width)/width)
            wrist_blend*=smooth((end+8-source)/8.)
            finger_blend=smooth((source-start+closing)/(closing+delay))*smooth((end+8-source)/8.)
            delta=(target_positions-positions)*wrist_blend[:,None]
            delta[:,2]+=arc*np.sin(np.pi*np.clip((source-start+24)/24,0.,1.))*(source<start)
            orientation=local_rotation_delta*wrist_blend[:,None]
            finger_delta=(desired_fingers-fingers)*finger_blend[:,None]
            raw[:,world,a:a+3]=delta/scale[a:a+3]
            raw[:,world,a+3:a+6]=orientation/scale[a+3:a+6]
            raw[:,world,a+6:a+6+len(fingers[0])]=finger_delta/scale[a+6:a+6+len(fingers[0])]
            offsets[world,list(hand.wrist_pos_q_ids)]=delta[0]
            q=hand.wrist_orient_q_id
            offsets[world,q:q+4]=(rotations[0]*Rotation.from_rotvec(orientation[0])).as_quat()-rotations[0].as_quat()
            offsets[world,list(hand.finger_q_ids)]=finger_delta[0]
    e.set_voc_scale(0.);e.reset(frame_id=0)
    e.reference_joint_q_offset.assign(offsets.reshape(-1));BaseEnv.reset(e,frame_id=0)
    initial=e.state_0.body_q.numpy().reshape(W,B,7).copy()
    for j in range(len(cfg['objects'])):
        assert np.allclose(initial[np.arange(W),object_ids[:,j]%B,:3],reference_pos[0,j],atol=1e-5)
    states=[]
    for i in range(N):
        assert np.all(e.timestep.numpy()==i)
        e.step(wp.array(raw[i].reshape(-1),device=e.device));assert np.all(e.voc_scale.numpy()==0.)
        states.append(e.state_0.body_q.numpy().reshape(W,B,7).copy())
    states=np.asarray(states);assert np.isfinite(states).all()
    objects=np.stack([states[:,np.arange(W),object_ids[:,j]%B] for j in range(len(cfg['objects']))],axis=2)
    distance=np.linalg.norm(objects[:,:,:,:3]-reference_pos[:,None],axis=-1)
    angle=2*np.arccos(np.clip(np.abs(np.sum(objects[:,:,:,3:]*reference_quat[:,None,:,[1,2,3,0]],axis=-1)),0.,1.))
    cost=np.mean(distance+.03*angle,axis=0)
    trials=[]
    for j,profile in enumerate(profiles):
        bad=(distance[:,j]>.1).any(axis=1)|(angle[:,j]>.5).any(axis=1)
        trials.append({'candidate':j,'profile':profile,'cost':float(cost[j].mean()),'violation_frames':int(bad.sum()),
            'objects':{n:{'mean_position_error_m':float(distance[:,j,b].mean()),'mean_orientation_error_rad':float(angle[:,j,b].mean()),
                'max_position_error_m':float(distance[:,j,b].max()),'max_orientation_error_rad':float(angle[:,j,b].max()),
                'cost':float(cost[j,b])} for b,n in enumerate(cfg['objects'])}})
    selected=[0]+[int(j) for j in np.argsort(cost[:len(profiles)].mean(axis=1))[:8] if j!=0]
    saved={'body_names':list(e.model.body_label[:B]),'reference_frames':np.arange(N),'all_object_poses':objects[:,:len(profiles)]}
    for j in selected:
        saved[f'candidate_{j}_body_poses']=states[:,j];saved[f'candidate_{j}_initial_body_poses']=initial[j]
    report={'episode':cfg['episode'],'diagnostic_only':True,'policy_inference':False,'object_assistance':0.,
        'full_native_control_frames':N,'reference':cfg['training_reference'],'source_object_poses_and_camera_unchanged':True,
        'robot_initial_pose_matches_tested_approach':True,'object_resets_within_attempt':False,
        'wrist_motor_profile':{'kp':250.,'kd':20.},'wrist_angular_kp':float(os.environ.get('V2D_DIAGNOSTIC_WRIST_ANGULAR_KP','25')),
        'fits':fits,'selected_candidates':selected,'trials':trials}
    (ROOT/f'evaluation/{LABEL}.json').write_text(json.dumps(report,indent=2))
    np.savez_compressed(ROOT/f'evaluation/{LABEL}.npz',**saved)
    print('Best complete approaches',[trials[j] for j in selected[1:4]],flush=True)
    raise SystemExit(0)

runner.run_training=probe
native['native']['main'].__wrapped__.__globals__['run_training']=probe
native['main']()
