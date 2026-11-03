"""Choose source posture examples from the hand's actual support phase."""


def template_frames(grasps,grasp):
    start,end=int(grasp['start']),int(grasp['end']);trim=min(10,(end-start)//4)
    interior=list(range(start+trim,end-trim+1))
    sole=[frame for frame in range(start,end+1) if not any(
        other['side']!=grasp['side'] and other['object']==grasp['object']
        and other['start']<=frame<=other['end'] for other in grasps)]
    if sole:
        preferred=[frame for frame in interior if frame in set(sole)]
        return preferred or sole,'sole_hand_support'
    return interior,'cooperative_support_only'
