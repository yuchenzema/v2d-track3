"""Diagnostic only: different finger corrections in each world, one learned actor."""
import json
import os
from pathlib import Path
import runpy
import numpy as np
from grip_bias_controller import install
install()
import flash_chord.evaluation.harness as harness
native_score=harness.score_rollout


def score(rollout,thresholds):
    profiles=json.loads(os.environ['V2D_GRIP_BIAS_PROFILES'])
    errors=np.asarray(rollout.tracking_error)
    limits=dict(zip(['wrist_position_m','wrist_orientation_rad','object_position_m','object_orientation_rad'],thresholds.as_tuple()))
    rows=[]
    for world,profile in enumerate(profiles):
        bad=np.zeros(len(errors),bool)
        for component,name in enumerate(rollout.tracking_error_names):
            if limits.get(name) is not None:bad|=errors[:,world,component]>limits[name]
        finite=bool(np.isfinite(rollout.achieved_pose_w[:,world]).all()) and world not in rollout.non_finite_worlds
        rows.append({'world':world,'profile':profile,'all_frames_finite':finite,
            'passed_all_tracking_frames':finite and not bool(bad.any()),
            'first_violation_control_frame':int(np.flatnonzero(bad)[0]) if bad.any() else None,
            'violation_frame_count':int(bad.sum()),
            'mean_tracking_errors':{name:float(errors[:,world,i].mean()) for i,name in enumerate(rollout.tracking_error_names)}})
    path=Path(os.environ['V2D_GRIP_SWEEP_OUTPUT'])
    path.write_text(json.dumps({'diagnostic_only':True,'one_different_controller_per_world':True,
        'not_a_32_world_success_rate_for_one_policy':True,'full_native_control_frames':len(errors),
        'profiles':rows,'object_assistance':0.,'failure_resets':False},indent=2))
    np.savez_compressed(path.with_suffix('.npz'),achieved_pose_w=rollout.achieved_pose_w,
        reference_pose_w=rollout.reference_pose_w,tracking_error=errors,
        object_body_names=rollout.object_body_names,tracking_error_names=rollout.tracking_error_names)
    return native_score(rollout,thresholds)


harness.score_rollout=score
runpy.run_path('/episode/scripts/evaluate_reference_controller.py',run_name='__main__')
