"""Cold-start, continuous reference-following diagnostic; never a learned policy."""
import json,os,runpy
from pathlib import Path
import numpy as np
from scipy.spatial.transform import Rotation
import warp as wp
import flash_chord.training.flash_sac.runner as runner

ROOT=Path('/episode')
cfg=json.loads((ROOT/'episode-config.json').read_text())
LABEL=os.environ.get('V2D_PICKUP_GATE_LABEL','cold-pickup-gate')
assert LABEL.replace('-','').isalnum()
os.environ['V2D_STATIC_GRASP_WORLD_COUNT']='32'
native=runpy.run_path('/episode/scripts/diagnose_static_grasp.py',run_name='static_probe_library')

def probe(env,*args,**kwargs):
    e=env.env;W=e.world_count;B=e.model.body_count//W
    assert not e.config.auto_reset and W==32
    e.set_voc_scale(0.);e.reset(frame_id=0);e.set_voc_scale(0.)
    indices=e.command.body_ids_w.numpy().reshape(W,len(cfg['objects']))%B
    right=next(h for h in e.scene.layout.hands if h.side=='right').palm_frame.body_id%B
    initial=e.state_0.body_q.numpy().reshape(W,B,7).copy()
    states=[];frames=[];poses=[];errors=[];angles=[]
    raw=wp.zeros(e.action_input.size,dtype=wp.float32,device=e.device)
    for _ in range(e.reference.num_frames-1):
        frame=int(e.timestep.numpy()[0]);assert frame<len(e.reference.object_body_pos_w())
        e.step(raw)
        assert np.all(e.voc_scale.numpy()==0.)
        current=e.state_0.body_q.numpy().reshape(W,B,7)
        assert np.isfinite(current).all()
        actual=current[np.arange(W),indices[:,0]]
        target=np.asarray(e.reference.object_body_pos_w()[frame,0])
        target_q=np.asarray(e.reference.object_body_quat_w()[frame,0])
        errors.append(np.linalg.norm(actual[:,:3]-target,axis=-1))
        angles.append((Rotation.from_quat(target_q,scalar_first=True).inv()*Rotation.from_quat(actual[:,3:])).magnitude())
        states.append(current[0].copy());poses.append(actual.copy())
        frames.append(frame*(cfg['frames']-1)/(e.reference.num_frames-1))
    frames=np.asarray(frames);poses=np.asarray(poses);errors=np.asarray(errors);angles=np.asarray(angles)
    held=(frames>=40)&(frames<=80)
    lifted=(poses[held,:,2]-initial[np.arange(W),indices[:,0],2][None,:]).max(axis=0)
    pickup=np.all(errors[held]<=.03,axis=0)&np.all(angles[held]<=.35,axis=0)&(lifted>=.015)
    full=(frames>=40)&(frames<=235)
    follows=np.all(errors[full]<=.03,axis=0)&np.all(angles[full]<=.35,axis=0)
    result={'episode':30,'diagnostic_only':True,'policy_inference':False,'initial_source_frame':0,
            'continuous_source_clock':True,'resets_after_start':False,'object_assistance':0.,
            'source_frames':frames.tolist(),'pickup_gate_passed':bool(pickup.all()),
            'source_timed_brush_follow_gate_passed':bool(follows.all()),'worlds':W,
            'worlds_with_physical_pickup':int(pickup.sum()),'worlds_with_full_brush_following':int(follows.sum()),
            'maximum_brush_lift_m':lifted.tolist(),'maximum_pickup_position_error_m':errors[held].max(axis=0).tolist(),
            'maximum_pickup_orientation_error_rad':angles[held].max(axis=0).tolist(),
            'limitation':'Reference-target controller diagnostic, not a trained policy; identical nominal resets do not establish policy robustness.'}
    (ROOT/f'evaluation/{LABEL}.json').write_text(json.dumps(result,indent=2))
    np.savez_compressed(ROOT/f'evaluation/{LABEL}.npz',body_poses=np.asarray(states),body_names=list(e.model.body_label[:B]),
                        object_poses=poses,source_frames=frames,position_error_m=errors,orientation_error_rad=angles)
    print(json.dumps({k:v for k,v in result.items() if k not in ['source_frames']},indent=2),flush=True)
    raise SystemExit(0)

runner.run_training=probe
native['native']['main'].__wrapped__.__globals__['run_training']=probe
native['main']()
