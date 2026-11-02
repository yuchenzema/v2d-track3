"""Strict, explicit local correspondence checks; never certify exact human imitation."""
import math
import statistics
import numpy as np
from scipy.spatial.transform import Rotation
CRITERIA_VERSION=3

LIMITS={'minimum_visible_pixels':30,'minimum_iou':.9,'maximum_centroid_error_px':3.,
        'maximum_position_error_m':.01,'maximum_orientation_error_rad':math.radians(3),
        'maximum_step_error_m':.005,'maximum_rotation_step_error_rad':math.radians(2)}

def intervals(indexes):
    result=[]
    for frame in sorted(set(indexes)):
        if result and frame==result[-1][1]+1:result[-1][1]=frame
        else:result.append([frame,frame])
    return result

def assess(summary,data):
    frames=data['frames'];count=summary['source_frames']
    assert data.get('version',summary['version'])==summary['version'],'Frame data belong to a different policy video'
    assert [f['frame'] for f in frames]==list(range(count))
    for i,frame in enumerate(frames):
        if 'time_s' in frame:assert abs(frame['time_s']-i/20)<1e-6,'Source timestamps changed'
        if 'recorded_control_frame' in frame:
            total=summary.get('native_control_frames',2*(count-1))
            assert frame['recorded_control_frame']==round(i*(total-1)/(count-1)),'Policy source-clock alignment changed'
    objects={};all_bad=set();unobserved=set();reconstruction_bad=set();policy_bad=set()
    for name in summary['objects']:
        failures=[];uncertain=[];ref_bad=[];errors=[]
        for frame in frames:
            i=frame['frame'];v=frame['objects'][name];a=v['rollout'];r=v['reconstruction']
            if a['visible_source_pixels']<LIMITS['minimum_visible_pixels']:
                uncertain.append(i);unobserved.add(i);continue
            reference_good=(r['iou']>=.8 and r['centroid_error_px'] is not None and r['centroid_error_px']<=8)
            step_error=rotation_step_error=0.
            if i:
                prev=frames[i-1]['objects'][name]
                step_error=float(np.linalg.norm((np.array(v['rollout_position_world_m'])-prev['rollout_position_world_m'])-(np.array(v['reconstruction_position_world_m'])-prev['reconstruction_position_world_m'])))
                if all(key in v and key in prev for key in ['rollout_orientation_world_xyzw','reconstruction_orientation_world_xyzw']):
                    actual=Rotation.from_quat(prev['rollout_orientation_world_xyzw']).inv()*Rotation.from_quat(v['rollout_orientation_world_xyzw'])
                    expected=Rotation.from_quat(prev['reconstruction_orientation_world_xyzw']).inv()*Rotation.from_quat(v['reconstruction_orientation_world_xyzw'])
                    rotation_step_error=float((actual.inv()*expected).magnitude())
            finite=all(math.isfinite(value) for value in [a['iou'],a['centroid_error_px'] if a['centroid_error_px'] is not None else math.inf,
                       v['object_position_error_to_video_reconstruction_m'],v['object_orientation_error_to_video_reconstruction_rad'],step_error,rotation_step_error])
            bad=(not finite or a['iou']<LIMITS['minimum_iou'] or a['centroid_error_px'] is None or a['centroid_error_px']>LIMITS['maximum_centroid_error_px']
                 or v['object_position_error_to_video_reconstruction_m']>LIMITS['maximum_position_error_m']
                 or v['object_orientation_error_to_video_reconstruction_rad']>LIMITS['maximum_orientation_error_rad']
                 or step_error>LIMITS['maximum_step_error_m'] or rotation_step_error>LIMITS['maximum_rotation_step_error_rad'])
            if not reference_good:ref_bad.append(i);reconstruction_bad.add(i)
            if bad:
                failures.append(i);all_bad.add(i)
                if reference_good:policy_bad.add(i)
            error={'frame':i,'iou':a['iou'],'centroid_error_px':a['centroid_error_px'],
                           'position_error_m':v['object_position_error_to_video_reconstruction_m'],
                           'orientation_error_rad':v['object_orientation_error_to_video_reconstruction_rad'],
                           'trajectory_step_error_m':step_error,'rotation_step_error_rad':rotation_step_error}
            errors.append({key:(value if value is None or math.isfinite(value) else None) for key,value in error.items()})
        objects[name]={'failed_frames':failures,'failed_intervals':intervals(failures),'unobservable_frames':uncertain,
                       'reconstruction_mismatch_frames':ref_bad,'per_frame_errors':errors,
                       'mean_iou':statistics.fmean(x['iou'] for x in errors if x['iou'] is not None) if any(x['iou'] is not None for x in errors) else None}
    collision=summary.get('collision_validation') or {}
    contact_failed=bool(collision.get('all_recorded_frames_checked') is True and collision.get('failed_frame_count',0)>0)
    contact_frames=[]
    if contact_failed:
        native_count=collision.get('recorded_frame_count',0)
        if native_count>1:
            contact_frames=sorted({round(i*(count-1)/(native_count-1)) for i in collision.get('failed_control_frames',[]) if 0<=i<native_count})
            all_bad.update(contact_frames)
    action=('visibility_review' if not all_bad and unobserved and not contact_failed else
            'reconstruct' if len(reconstruction_bad)>max(5,count*.1) else 'train_rollout')
    windows=[]
    for start in range(0,count,10):
        end=min(count-1,start+19);bad=sum(i in all_bad for i in range(start,end+1))
        if bad:windows.append({'start':start,'end':end,'failed_frames':bad})
    windows.sort(key=lambda w:(w['failed_frames'],-w['start']),reverse=True)
    return {'episode':summary['episode'],'version':summary['version'],'criteria_version':CRITERIA_VERSION,'source_frames':count,'thresholds':LIMITS,
            'pixel_coordinate_resolution':[512,384],'source_frame_rate_hz':20,
            'all_frames_checked':True,'source_time_warping_used':False,'objects':objects,
            'failed_frame_count':len(all_bad),'unobservable_frame_count':len(unobserved),
            'reconstruction_mismatch_frame_count':len(reconstruction_bad),'policy_mismatch_with_good_reference_frame_count':len(policy_bad),
            'worst_windows':windows[:8],'recommended_action':action,
            'passes_strict_local_checks':not all_bad and not unobserved and not contact_failed,
            'exact_human_match_verified':False,
            'states_sha256':summary.get('recorded_states_sha256'),
            'original_video_sha256':summary.get('source_video_sha256'),
            'collision_validation':summary.get('collision_validation'),
            'contact_failure_source_frames':contact_frames,
            'contact_validation_failed':contact_failed,
            'limitation':'These are explicit local tolerances, not exact equivalence. 3D pose errors are measured against video reconstruction, not confidential human ground truth. Occluded frames remain unverified.'}
