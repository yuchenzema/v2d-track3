"""Fail-closed episode completion; tolerance checks never establish exactness."""
from rollout_collision_validation import collision_verified

def completion(check):
    reasons=[]
    if check.get('experimental_only'):
        reasons.append('Experimental training is not approved for strict episode completion.')
    if check.get('post_training_grasp_acceptance_required'):
        grasp=check.get('post_training_grasp_acceptance') or {};binding=check.get('official_render_binding') or {}
        if not (grasp.get('all_grasps_passed') is True and grasp.get('policy_inference') is True
                and grasp.get('checkpoint_sha256')==check.get('checkpoint_sha256') and check.get('checkpoint_sha256')
                and grasp.get('official_trajectory_sha256')==binding.get('official_trajectory_sha256')
                and binding.get('rendered_state_sha256')==check.get('states_sha256') and check.get('states_sha256')
                and binding.get('rendered_objects_match_official_recording') is True):
            reasons.append('Trained-policy grasp acceptance is missing, failed or bound to another rollout.')
    count=check.get('source_frames')
    expected_frames=(count if check.get('source_frame_playback') is True else 2*(count-1)) if isinstance(count,int) and count>1 else 0
    if not collision_verified(check.get('collision_validation') or {},check.get('states_sha256'),
                              check.get('version'),expected_frames):
        reasons.append('Penetration-free hand/object/environment contact has not been verified for this recorded rollout.')
    if not check.get('all_frames_checked'):reasons.append('Not every source frame has been checked.')
    if not check.get('passes_strict_local_checks'):reasons.append('Strict local motion checks have not passed.')
    if check.get('failed_frame_count')!=0:reasons.append('Object-motion discrepancies remain or their count is unknown.')
    if check.get('unobservable_frame_count')!=0:reasons.append('Some motion remains unverified or unobservable.')
    if check.get('source_time_warping_used') is not False:reasons.append('Preservation of source timing has not been verified.')
    if check.get('exact_human_match_verified') is not True:
        reasons.append('Exact human-object motion equality has not been verified; local tolerances are insufficient.')
    evidence=check.get('exact_match_evidence',{})
    bound=bool(evidence.get('version') and evidence.get('version')==check.get('version')
               and evidence.get('states_sha256') and evidence.get('states_sha256')==check.get('states_sha256')
               and evidence.get('all_objects_all_frames_verified') is True
               and isinstance(check.get('source_frames'),int) and check['source_frames']>0
               and evidence.get('source_frame_count')==check['source_frames']
               and evidence.get('source_video_sha256') and evidence.get('source_video_sha256')==check.get('original_video_sha256')
               and evidence.get('method'))
    if not bound:reasons.append('Independent exact-match evidence is missing or belongs to another rollout.')
    return {'episode_done':not reasons,'advance_allowed':not reasons,'blocking_reasons':reasons}

def first_unfinished(order,checks):
    return next((ep for ep in order if not completion(checks.get(str(ep),{}))['episode_done']),None)
