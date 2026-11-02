"""Diagnostic search for physical hand grasps; no learned-policy claim."""
import itertools
import json
import os
from pathlib import Path
import runpy
import numpy as np
from scipy.spatial.transform import Rotation
import trimesh
import warp as wp
import flash_chord.training.flash_sac.runner as runner

ROOT=Path('/episode');cfg=json.loads((ROOT/'episode-config.json').read_text())
profiles=[{'side':'both','finger_rad':0.,'inward_m':0.}]
profiles += [{'side':side,'finger_rad':finger,'inward_m':distance}
    for side in ['left','right'] for finger,distance in itertools.product([0.,.08,.16,.24,.32],[-.01,0.,.01,.02,.03])]
os.environ['V2D_STATIC_GRASP_WORLD_COUNT']=str(len(profiles))
os.environ['V2D_WRIST_MOTOR_PROFILE']=json.dumps({'kp':250.,'kd':20.})
native=runpy.run_path('/episode/scripts/diagnose_static_grasp.py',run_name='static_probe_library')


def probe(env,*args,**kwargs):
    e=env.env;W=e.world_count;B=e.model.body_count//W
    assert W==len(profiles) and not e.config.auto_reset
    source_frame=native['source_frame'];frame=int(round(source_frame*(e.reference.num_frames-1)/(cfg['frames']-1)))
    e.set_voc_scale(0.);e.reset(frame_id=frame);e.set_voc_scale(0.)
    initial=e.state_0.body_q.numpy().copy();names=list(e.model.body_label)
    layout=e.action.buffers.layout;scale=e.action.buffers.scale.numpy()
    object_ids=e.command.body_ids_w.numpy().reshape(W,len(cfg['objects']))
    hands={h.side:h for h in e.scene.layout.hands}
    base=np.zeros((W,layout.action_dim),np.float32);deltas={}
    position=np.asarray(e.reference.object_body_pos_w())[frame]
    rotations=Rotation.from_quat(np.asarray(e.reference.object_body_quat_w())[frame],scalar_first=True).as_matrix()
    centers=[trimesh.load(ROOT/f'reconstruction/mesh_priors/{name}.ply',process=False).vertices.mean(0) for name in cfg['objects']]
    for side,start,count in zip(layout.sides,layout.hand_action_starts,layout.finger_counts):
        grasp=next(g for g in cfg['grasps'] if g['side']==side and g['start']<=source_frame<=g['end'])
        body=grasp['object'];wrist=np.asarray(e.reference.wrist_pos_w(side))[frame]
        direction=rotations[body]@centers[body]+position[body]-wrist;direction/=np.linalg.norm(direction)
        deltas[side]={'object':body,'world_inward_direction':direction.tolist(),
            'object_local_inward_direction':(rotations[body].T@direction).tolist()}
        for world,profile in enumerate(profiles):
            if profile['side'] not in [side,'both']:continue
            base[world,start:start+3]=direction*profile['inward_m']/scale[start:start+3]
            for j,name in enumerate(hands[side].finger_joint_names):
                if name.endswith(('_FE','_IP','_PIP','_DIP')):base[world,start+6+j]=profile['finger_rad']/scale[start+6+j]
    poses=[]
    for step in range(60):
        e.timestep.fill_(frame);lift=.03*float(np.clip((step-19)/20.,0.,1.));raw=base.copy()
        for start in layout.hand_action_starts:raw[:,start+2]+=lift/scale[start+2]
        e.step(wp.array(raw.reshape(-1),device=e.device));assert np.all(e.voc_scale.numpy()==0.)
        poses.append(e.state_0.body_q.numpy().copy())
    states=np.asarray(poses);assert np.isfinite(states).all();rows=[]
    for world,profile in enumerate(profiles):
        objects={}
        for body,name in enumerate(cfg['objects']):
            side=next(g['side'] for g in cfg['grasps'] if g['object']==body and g['start']<=source_frame<=g['end'])
            oi=int(object_ids[world,body]);wi=hands[side].palm_frame.body_id+world*B
            assert wi<e.model.body_count
            wrist=states[-1,wi,:3]-states[19,wi,:3];actual=states[:,oi];p0=initial[oi]
            drift=float(np.linalg.norm(actual[19,:3]-p0[:3]))
            follow=float(np.linalg.norm(actual[-1,:3]-(actual[19,:3]+wrist)))
            angle=float((Rotation.from_quat(p0[3:]).inv()*Rotation.from_quat(actual[-1,3:])).magnitude())
            stable=drift<=.015 and follow<=.02 and angle<=.3 and wrist[2]>=.02
            cost=drift+follow+.03*angle+max(0.,.02-float(wrist[2]))
            objects[name]={'side':side,'static_position_drift_m':drift,'object_following_error_m':follow,
                'object_orientation_change_rad':angle,'actual_hand_lift_m':float(wrist[2]),
                'stable_hold_and_lift':bool(stable),'diagnostic_cost':cost}
        rows.append({'world':world,'profile':profile,'objects':objects})
    report={'episode':cfg['episode'],'diagnostic_only':True,'policy_inference':False,
        'source_frame':source_frame,'wrist_motor_profile':{'kp':250.,'kd':20.,'force_limit_N':45.},
        'object_assistance':0.,'object_state_resets_during_probe':False,
        'different_grasp_candidate_per_world':True,'profiles':rows,'grasp_directions':deltas,
        'full_source_timing_test_still_required':True}
    output=ROOT/'evaluation/grasp-offset-grid.json';output.write_text(json.dumps(report,indent=2))
    np.savez_compressed(output.with_suffix('.npz'),body_poses=states,initial_body_poses=initial,
        body_names=names,object_body_ids=object_ids,diagnostic_kind=__doc__)
    for name in cfg['objects']:
        best=min(rows,key=lambda row:row['objects'][name]['diagnostic_cost'])
        print(name,best['profile'],best['objects'][name],flush=True)
    raise SystemExit(0)


runner.run_training=probe
# The native entry point imported its runner while the static library loaded.
# Update that actual function's globals as well as the runner module binding.
native['native']['main'].__wrapped__.__globals__['run_training']=probe
native['main']()
