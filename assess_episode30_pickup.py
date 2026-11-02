"""Reject learned rollouts that move a hand without picking up and holding the brush."""
import argparse,json
from pathlib import Path
import numpy as np
from scipy.spatial.transform import Rotation

def assess_arrays(objects,palms,source_frames,reference_positions,assistance):
    finite=bool(np.isfinite(objects).all() and np.isfinite(palms).all())
    if not finite:return {'passes':False,'reason':'Nonfinite physical states'}
    held=(source_frames>=55)&(source_frames<=235)
    pickup=(source_frames>=55)&(source_frames<=80)
    if held.sum()<2 or pickup.sum()<2:return {'passes':False,'reason':'Pickup or held-motion frames missing'}
    lift=float((objects[pickup,2]-objects[0,2]).max())
    relative=Rotation.from_quat(palms[held,3:]).inv().apply(objects[held,:3]-palms[held,:3])
    drift=float(np.linalg.norm(relative-np.median(relative,axis=0),axis=-1).max())
    errors=np.linalg.norm(objects[held,:3]-reference_positions[held],axis=-1)
    zero_assistance=bool(np.all(np.asarray(assistance)==0))
    checks={'brush_lift_at_least_15mm':lift>=.015,'hand_relative_grip_drift_at_most_20mm':drift<=.02,
            'held_position_error_at_most_50mm':bool(errors.max()<=.05),'zero_object_assistance':zero_assistance}
    return {'passes':all(checks.values()),'checks':checks,'actual_brush_lift_m':lift,
            'maximum_hand_relative_grip_drift_m':drift,'maximum_held_position_error_m':float(errors.max()),
            'mean_held_position_error_m':float(errors.mean()),'frames_checked':len(objects),
            'exact_human_motion_match_verified':False,
            'limitation':'Pickup prerequisite only. Inferred 3D reference, silhouette, orientation and full timing still require separate source-video checks.'}

def assess_file(run,states):
    import pyarrow.parquet as pq
    cfg=json.loads((run/'episode-config.json').read_text());data=np.load(states)
    names=data['body_names'].tolist();poses=data['body_poses'];frames=data['reference_frames']
    if not (len(frames)==2*(cfg['frames']-1) and frames[0]==0 and np.all(np.diff(frames)==1)):
        return {'passes':False,'reason':'Missing, reset or shifted source clock in recorded policy states'}
    source=frames*(cfg['frames']-1)/(len(frames)-1)
    row=pq.read_table(run/f"human_motion_data/{cfg['training_reference']}/processed/sequence_id=episode_000030/robot_name=sharpa_wave/motion.parquet").to_pylist()[0]
    ref=np.asarray(row['object_body_position'])[:,0]
    expected=np.stack([np.interp(source,np.arange(len(ref)),ref[:,axis]) for axis in range(3)],axis=-1)
    return assess_arrays(poses[:,names.index('track3_mini_sweeper/object')],
                         poses[:,names.index('right_sharpa_wave/right_hand_C_MC')],source,expected,data['object_assistance'])

if __name__=='__main__':
    from improvement_common import save,utc
    p=argparse.ArgumentParser();p.add_argument('--run',type=Path,required=True);p.add_argument('--states',type=Path,required=True);p.add_argument('--output',type=Path,required=True);a=p.parse_args()
    result=assess_file(a.run,a.states);result.update(updated_utc=utc(),states=str(a.states));save(a.output,result);print(json.dumps(result,indent=2))
