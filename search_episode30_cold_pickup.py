"""Optimize hand motor commands through real approach and pickup from frame zero."""
import json,os,runpy
from pathlib import Path
import numpy as np
from scipy.spatial.transform import Rotation
import warp as wp
import flash_chord.training.flash_sac.runner as runner

ROOT=Path('/episode');cfg=json.loads((ROOT/'episode-config.json').read_text())
W=128;os.environ['V2D_STATIC_GRASP_WORLD_COUNT']=str(W)
PROFILE=json.loads(os.environ.get('V2D_COLD_PICKUP_PROFILE','{"kp":250,"kd":20}'))
os.environ['V2D_WRIST_MOTOR_PROFILE']=json.dumps(PROFILE)
REFERENCE=os.environ.get('V2D_COLD_PICKUP_REFERENCE','track3_pickup_source_hand')
os.environ['V2D_GRASP_PROBE_REFERENCE']=REFERENCE
native=runpy.run_path('/episode/scripts/diagnose_static_grasp.py',run_name='static_probe_library')

def probe(env,*args,**kwargs):
    e=env.env;B=e.model.body_count//W;assert e.world_count==W and not e.config.auto_reset
    hand=next(h for h in e.scene.layout.hands if h.side=='right');palm=hand.palm_frame.body_id%B
    layout=e.action.buffers.layout;start=layout.hand_action_starts[layout.sides.index('right')];scale=e.action.buffers.scale.numpy()
    ids=e.command.body_ids_w.numpy().reshape(W,len(cfg['objects']))%B
    refnames=e.reference.finger_joint_names('right');order=[refnames.index(n) for n in hand.finger_joint_names]
    fingers=np.asarray(e.reference.finger_joint_pos('right'))[:,order]
    limits_low=np.asarray(e.model.joint_limit_lower.numpy())[list(hand.finger_dof_ids)]+1e-6
    limits_high=np.asarray(e.model.joint_limit_upper.numpy())[list(hand.finger_dof_ids)]-1e-6
    frame34=round(34*(e.reference.num_frames-1)/(cfg['frames']-1))
    closed=fingers[frame34].copy();opened=closed.copy()
    for j,name in enumerate(hand.finger_joint_names):
        if name.endswith(('_MCP_FE','_PIP','_DIP','_IP')):opened[j]=np.clip(0.,limits_low[j],limits_high[j])
    F=9+len(closed)
    left=next(h for h in e.scene.layout.hands if h.side=='left');left_start=layout.hand_action_starts[layout.sides.index('left')]
    left_names=e.reference.finger_joint_names('left');left_order=[left_names.index(n) for n in left.finger_joint_names]
    left_fingers=np.asarray(e.reference.finger_joint_pos('left'))[:,left_order]
    left_flex=np.array([name.endswith(('_MCP_FE','_PIP','_DIP','_IP')) for name in left.finger_joint_names])
    left_lo=np.asarray(e.model.joint_limit_lower.numpy())[list(left.finger_dof_ids)]+1e-6
    left_hi=np.asarray(e.model.joint_limit_upper.numpy())[list(left.finger_dof_ids)]-1e-6
    # Right approach/grip plus left rack grip; all changes are motor targets.
    lo=np.r_[[-.035]*3,[-.35]*3,27.,4.,0.,limits_low,[-.03]*3,.6]
    hi=np.r_[[.035]*3,[.35]*3,40.,12.,.03,limits_high,[.03]*3,1.6]
    offset=np.asarray(e.reference.object_body_pos_w()[0,0])-np.asarray(e.reference.object_body_pos_w()[frame34,0])
    center=np.r_[np.clip(offset,-.025,.025),np.zeros(3),33.,7.,.01,closed,np.zeros(3),1.]
    if os.environ.get('V2D_COLD_PICKUP_SEED'):
        saved=json.loads((ROOT/os.environ['V2D_COLD_PICKUP_SEED']).read_text())['best']['parameters']
        if len(saved)==F:saved+=[0.,0.,0.,1.]
        assert len(saved)==len(center);center=np.clip(saved,lo,hi)
    std=np.r_[[.008]*3,[.08]*3,3.,2.,.006,[.13]*len(closed),[.01]*3,.2]
    rng=np.random.default_rng(2026102630+int(os.environ.get('V2D_COLD_PICKUP_ROUND','0')));best=None;best_values=None;best_record=None;history=[]
    generations=int(os.environ.get('V2D_COLD_PICKUP_GENERATIONS','8'));assert 1<=generations<=24
    for generation in range(generations):
        values=np.clip(rng.normal(center,std,(W,len(center))),lo,hi);values[0]=np.clip(center,lo,hi)
        if best_values is not None:values[1]=best_values
        e.set_voc_scale(0.);e.reset(frame_id=0);e.set_voc_scale(0.)
        initial=e.state_0.body_q.numpy().reshape(W,B,7).copy()
        trajectory=[];targets=[];sources=[];controls=[];palms=[];all_object_poses=[]
        native_steps=round(85*(e.reference.num_frames-1)/(cfg['frames']-1))
        for _ in range(native_steps):
            frame=int(e.timestep.numpy()[0]);source=frame*(cfg['frames']-1)/(e.reference.num_frames-1)
            raw=np.zeros((W,layout.action_dim),np.float32)
            blend=float(np.clip(min((source-20)/10.,(253-source)/8.),0.,1.))
            raw[:,start:start+3]=blend*values[:,:3]/scale[start:start+3]
            raw[:,start:start+3][:,2]+=values[:,8]*max(0.,1-abs(source-25)/9.)/scale[start+2]
            raw[:,start+3:start+6]=blend*values[:,3:6]/scale[start+3:start+6]
            closing=np.clip((source-(values[:,6]-values[:,7]/2))/values[:,7],0.,1.)
            desired=(1-closing[:,None])*opened+closing[:,None]*values[:,9:F]
            raw[:,start+6:start+6+len(closed)]=(desired-fingers[frame])/scale[start+6:start+6+len(closed)]
            left_blend=float(np.clip(min((source-8)/6.,(253-source)/6.),0.,1.))
            raw[:,left_start:left_start+3]=left_blend*values[:,F:F+3]/scale[left_start:left_start+3]
            left_goal=np.tile(left_fingers[frame],(W,1))
            left_goal[:,left_flex]*=1+left_blend*(values[:,F+3,None]-1)
            left_goal=np.clip(left_goal,left_lo,left_hi)
            raw[:,left_start+6:left_start+6+len(left.finger_joint_names)]=(left_goal-left_fingers[frame])/scale[left_start+6:left_start+6+len(left.finger_joint_names)]
            e.step(wp.array(raw.reshape(-1),device=e.device));assert np.all(e.voc_scale.numpy()==0.)
            state=e.state_0.body_q.numpy().reshape(W,B,7)
            trajectory.append(state.copy());all_object_poses.append(state[np.arange(W)[:,None],ids].copy())
            targets.append(np.asarray(e.reference.object_body_pos_w()[frame]).copy());sources.append(source)
            palms.append(state[:,palm].copy());controls.append(raw.copy())
        sources=np.asarray(sources);objects=np.asarray(all_object_poses);targets=np.asarray(targets);palms=np.asarray(palms)
        finite=np.isfinite(objects).all(axis=(0,2,3))&np.isfinite(palms).all(axis=(0,2))
        errors=np.linalg.norm(objects[:,:,:,:3]-targets[:,None,:,:],axis=-1)
        held=(sources>=55)&(sources<=80)
        lift=(objects[held,:,0,2]-initial[np.arange(W),ids[:,0],2]).max(axis=0)
        relative=np.full((held.sum(),W,3),np.nan)
        for t,index in enumerate(np.flatnonzero(held)):
            good=np.flatnonzero(finite)
            relative[t,good]=Rotation.from_quat(palms[index,good,3:]).inv().apply(objects[index,good,0,:3]-palms[index,good,:3])
        following=np.linalg.norm(relative-relative[:1],axis=-1).max(axis=0)
        mean=errors[held,:,0].mean(axis=0);maximum=errors[held,:,0].max(axis=0)
        rack_error=errors[:,:,1].max(axis=0)
        stable=finite&(lift>=.02)&(following<=.015)&(mean<=.03)&(maximum<=.05)&(rack_error<=.05)
        cost=2*mean+maximum+2*following+3*np.maximum(0.,.02-lift)+rack_error
        cost[~finite]=np.inf
        order=np.lexsort((cost,~stable));j=int(order[0]);rank=(not bool(stable[j]),float(cost[j]))
        if best is None or rank<best:
            best=rank;best_values=values[j].copy()
            best_record={'generation':generation,'world':j,'cold_pickup_verified':bool(stable[j]),
                         'actual_object_lift_m':float(lift[j]),'hand_relative_grip_drift_m':float(following[j]),
                         'mean_held_object_position_error_m':float(mean[j]),'maximum_held_object_position_error_m':float(maximum[j]),
                         'maximum_rack_position_error_m':float(rack_error[j]),'cost':float(cost[j]),'parameters':best_values.tolist()}
            # Save the winning world's actual bodies, not world0 or a posed animation.
            winning_objects=objects[:,j].copy();winning_palms=palms[:,j].copy();winning_controls=np.asarray(controls)[:,j].copy()
            np.savez_compressed(ROOT/'evaluation/cold-pickup-search-best.npz',source_frames=sources,object_poses=winning_objects,
                                palm_poses=winning_palms,actions=winning_controls,reference_positions=targets,
                                body_poses=np.asarray(trajectory)[:,j],body_names=list(e.model.body_label[:B]),initial_body_poses=initial[j])
        history.append({'generation':generation,'verified_cold_pickups':int(stable.sum()),'finite_worlds':int(finite.sum()),'best':best_record})
        report={'episode':30,'method':__doc__,'diagnostic_only':True,'policy_inference':False,'initial_source_frame':0,
                'resets_after_start':False,'object_assistance':0.,'objects_moved_by_hand_contact_only':True,
                'reference':REFERENCE,'world_count':W,'best':best_record,'history':history,
                'wrist_motor_profile':PROFILE,'parameter_layout':{'right_finger_end':F,'left_position_start':F,'left_flexion_scale':F+3},
                'left_finger_limits':{'lower':left_lo.tolist(),'upper':left_hi.tolist()},
                'left_finger_names':list(left.finger_joint_names),
                'search_complete':bool(stable[j] or generation==generations-1),
                'full_source_motion_verified':False,'pickup_is_not_full_episode_success':True}
        (ROOT/'evaluation/cold-pickup-search.json').write_text(json.dumps(report,indent=2))
        print('Cold pickup generation',generation,'stable',int(stable.sum()),'best',best_record,flush=True)
        if stable[j]:break
        elite=values[[k for k in order if finite[k]][:16]];assert len(elite)
        center=.25*center+.75*elite.mean(axis=0)
        std=np.maximum(.6*std+.4*elite.std(axis=0),np.r_[[.002]*3,[.025]*3,1.,.5,.002,[.035]*len(closed),[.002]*3,.05])
    raise SystemExit(0)

runner.run_training=probe
native['native']['main'].__wrapped__.__globals__['run_training']=probe
native['main']()
