"""Diagnostic full-motion grasp search. All objects move under unassisted physics."""
import itertools
import json
import os
from pathlib import Path
import runpy
import numpy as np
from scipy.spatial.transform import Rotation
import warp as wp
import flash_chord.training.flash_sac.runner as runner

ROOT=Path('/episode')
cfg=json.loads((ROOT/'episode-config.json').read_text())
prior=json.loads((ROOT/'evaluation/grasp-pose-search.json').read_text())
WORLDS=128
os.environ['V2D_STATIC_GRASP_WORLD_COUNT']=str(WORLDS)
os.environ['V2D_WRIST_MOTOR_PROFILE']=json.dumps({'kp':250.,'kd':20.})
native=runpy.run_path('/episode/scripts/diagnose_static_grasp.py',run_name='static_probe_library')


def smooth(x):
    x=np.clip(x,0.,1.)
    return x*x*(3.-2.*x)


def probe(env,*args,**kwargs):
    e=env.env;W=e.world_count;B=e.model.body_count//W
    assert W==WORLDS and not e.config.auto_reset
    layout=e.action.buffers.layout;scales=e.action.buffers.scale.numpy()
    object_ids=e.command.body_ids_w.numpy().reshape(W,len(cfg['objects']))
    reference_pos=np.asarray(e.reference.object_body_pos_w())
    reference_quat=np.asarray(e.reference.object_body_quat_w())
    reference_rot=Rotation.from_quat(reference_quat.reshape(-1,4),scalar_first=True).as_matrix().reshape(*reference_quat.shape[:2],3,3)
    N=e.reference.num_frames
    proposals=[None]+list(itertools.product([0.,.015,.03],[-.015,0.,.015],[-.015,0.,.015],[.5,1.],[0,4]))
    assert len(proposals)<=W
    padded=proposals+[None]*(W-len(proposals))
    report={'episode':cfg['episode'],'diagnostic_only':True,'policy_inference':False,
            'reference':cfg['training_reference'],'object_assistance':0.,'resets_within_attempt':False,
            'source_object_poses_and_camera_unchanged':True,'wrist_motor_profile':{'kp':250.,'kd':20.},
            'full_native_control_frames':N,'searches':{}}
    saved={'body_names':list(e.model.body_label[:B]),'reference_frames':np.arange(N)}
    for body,name in enumerate(cfg['objects']):
        if cfg['episode']==30 and name!='mini_sweeper':continue
        best=prior['best_candidates'][name];side=best['side']
        grasp=next(g for g in cfg['grasps'] if g['side']==side and g['object']==body)
        start,end=grasp['start'],grasp['end']
        hand=next(h for h in e.scene.layout.hands if h.side==side)
        action_start=layout.hand_action_starts[layout.sides.index(side)]
        names=e.reference.finger_joint_names(side);idx=[names.index(n) for n in hand.finger_joint_names]
        assert best['finger_joint_names']==list(hand.finger_joint_names)
        original=np.asarray(e.reference.finger_joint_pos(side))[:,idx]
        desired=np.asarray(best['finger_joints'])
        frame=round(prior['source_frame']*(N-1)/(cfg['frames']-1))
        inward=reference_rot[frame,body].T@np.asarray(best['world_inward_direction'])
        vertical=reference_rot[frame,body].T@np.array([0.,0.,1.])
        lateral=np.cross(vertical,inward);lateral/=max(np.linalg.norm(lateral),1e-9)
        e.set_voc_scale(0.);e.reset(frame_id=0);initial=e.state_0.body_q.numpy().reshape(W,B,7).copy()
        states=[]
        for i in range(N):
            assert np.all(e.timestep.numpy()==i)
            source=i*(cfg['frames']-1)/(N-1)
            raw=np.zeros((W,layout.action_dim),np.float32)
            for world,proposal in enumerate(padded):
                if proposal is None:continue
                distance,z,tangent,alpha,delay=proposal
                fade=smooth((end+8-source)/8.)
                wrist_blend=smooth((source-start+8)/8.)*fade
                finger_blend=smooth((source-start+8)/(8.+delay))*fade
                local=inward*distance+vertical*z+lateral*tangent
                raw[world,action_start:action_start+3]=wrist_blend*(reference_rot[i,body]@local)/scales[action_start:action_start+3]
                a=action_start+6;b=a+len(desired)
                raw[world,a:b]=finger_blend*alpha*(desired-original[i])/scales[a:b]
            e.step(wp.array(raw.reshape(-1),device=e.device))
            assert np.all(e.voc_scale.numpy()==0.)
            states.append(e.state_0.body_q.numpy().reshape(W,B,7).copy())
        states=np.asarray(states)
        assert np.isfinite(states).all()
        objects=np.stack([states[:,np.arange(W),object_ids[:,j]%B] for j in range(len(cfg['objects']))],axis=2)
        position=np.linalg.norm(objects[:,:,:,:3]-reference_pos[:,None,:,:],axis=-1)
        dot=np.abs(np.sum(objects[:,:,:,3:]*reference_quat[:,None,:,[1,2,3,0]],axis=-1))
        angle=2*np.arccos(np.clip(dot,0.,1.))
        cost=np.mean(position+.03*angle,axis=0)
        rows=[]
        for j,proposal in enumerate(proposals):
            rows.append({'candidate':j,'parameters':proposal,'target_cost':float(cost[j,body]),
                'objects':{n:{'mean_position_error_m':float(position[:,j,k].mean()),
                              'mean_orientation_error_rad':float(angle[:,j,k].mean()),
                              'cost':float(cost[j,k])} for k,n in enumerate(cfg['objects'])}})
        selected=[0]+[j for j in np.argsort(cost[:len(proposals),body])[:6] if j!=0]
        report['searches'][name]={'side':side,'object_index':body,'grasp':grasp,
            'basis_object_local':{'inward':inward.tolist(),'vertical':vertical.tolist(),'lateral':lateral.tolist()},
            'finger_joint_names':list(hand.finger_joint_names),'finger_target':desired.tolist(),
            'selected_candidates':list(map(int,selected)),'trials':rows}
        for j in selected:
            saved[f'{name}_candidate_{j}_body_poses']=states[:,j]
            saved[f'{name}_candidate_{j}_initial_body_poses']=initial[j]
        saved[name+'_all_object_poses']=objects[:,:len(proposals)]
        print(name,'baseline',rows[0],'best',[rows[j] for j in selected[1:3]],flush=True)
    (ROOT/'evaluation/dynamic-grasp-search.json').write_text(json.dumps(report,indent=2))
    np.savez_compressed(ROOT/'evaluation/dynamic-grasp-search.npz',**saved)
    raise SystemExit(0)


runner.run_training=probe
native['native']['main'].__wrapped__.__globals__['run_training']=probe
native['main']()
