"""Assign anatomical hand identity by votes over complete continuous tracks."""
from pathlib import Path
import json
import cv2
import numpy as np
from scipy.optimize import linear_sum_assignment
from improvement_common import save,utc,digest


def decide(votes,observations):
    votes=np.asarray(votes,float);observations=np.asarray(observations)
    if np.any(observations<10):raise RuntimeError('Insufficient full-track handedness observations')
    identity=float(votes[0,0]+votes[1,1]);swapped=float(votes[0,1]+votes[1,0]);track_for_side=[0,1] if identity>=swapped else [1,0]
    confidence=[float(votes[track,side]/max(1e-12,votes[track].sum())) for side,track in enumerate(track_for_side)]
    if min(confidence)<.6:raise RuntimeError('Whole-trajectory anatomical hand vote is ambiguous')
    return track_for_side,confidence


def decide_observations(votes,observations,rows):
    """Uncertain grasp detections must not dilute separated strong anchors."""
    try:
        permutation,confidence=decide(votes,observations)
        return permutation,confidence,dict(method='all_matched_observations')
    except RuntimeError as original:
        strong=np.zeros((2,2));counts=np.zeros(2,int);anchors={0:[],1:[]}
        for row in rows:
            if row['confidence']<.8:continue
            track=row['track'];side=int(row['label']=='Right');score=row['confidence']
            strong[track,side]+=score;strong[track,1-side]+=1-score;counts[track]+=1;anchors[track].append(row)
        try:permutation,confidence=decide(strong,counts)
        except RuntimeError:raise original
        for side,track in enumerate(permutation):
            windows={}
            for row in anchors[track]:windows.setdefault(row['frame']//20,[]).append(row)
            supported=[]
            for window,samples in windows.items():
                probability=np.mean([r['confidence'] if int(r['label']=='Right')==side else 1-r['confidence'] for r in samples])
                if len(samples)>=3 and probability>=.8:supported.append(window)
            if len(supported)<2:raise original
        return permutation,confidence,dict(method='strong_observations_in_separated_source_windows',minimum_label_confidence=.8,
             minimum_observations_per_track=10,minimum_supported_windows_per_track=2,window_frames=20,
             votes=strong.tolist(),matched_observations=counts.tolist(),all_observation_vote_error=str(original))


def run(root):
    import mediapipe as mp
    from mediapipe.tasks.python import BaseOptions
    from mediapipe.tasks.python.vision import HandLandmarker,HandLandmarkerOptions,RunningMode
    root=Path(root);rec=root/'reconstruction';source=rec/'hand_measurements.npz';report=rec/'global-hand-identity.json'
    if report.exists() and (rec/'hand_measurements_anatomical.npz').exists():
        previous=json.loads(report.read_text())
        if previous.get('status')!='failed' and previous.get('original_hand_measurements_sha256')==digest(source):return
    cfg=json.loads((root/'episode-config.json').read_text());old=np.load(source);values={k:old[k].copy() for k in old.files}
    votes=np.zeros((2,2));observations=np.zeros(2,int);rows=[]
    options=HandLandmarkerOptions(base_options=BaseOptions(model_asset_path=str(root/'models/hand_landmarker.task')),
        running_mode=RunningMode.VIDEO,num_hands=2,min_hand_detection_confidence=.35,min_hand_presence_confidence=.35,min_tracking_confidence=.35)
    with HandLandmarker.create_from_options(options) as detector:
        for frame in range(cfg['frames']):
            image=cv2.imread(str(rec/f'frames/{frame:06d}.png'));height,width=image.shape[:2]
            result=detector.detect_for_video(mp.Image(image_format=mp.ImageFormat.SRGB,data=cv2.cvtColor(image,cv2.COLOR_BGR2RGB)),round(frame*1000/cfg['fps']))
            if not result.hand_landmarks:continue
            points=np.array([[[p.x*width,p.y*height] for p in landmarks] for landmarks in result.hand_landmarks])
            cost=np.full((len(points),2),1e6)
            for track in range(2):
                if values['observed'][frame,track]:cost[:,track]=np.median(np.linalg.norm(points-values['uv'][frame,track],axis=2),axis=1)
            detections,tracks=linear_sum_assignment(cost)
            for detection,track in zip(detections,tracks):
                if cost[detection,track]>35:continue
                category=result.handedness[detection][0]
                if category.category_name not in {'Left','Right'}:continue
                side=int(category.category_name=='Right');score=float(category.score)
                votes[track,side]+=score;votes[track,1-side]+=1-score;observations[track]+=1
                rows.append(dict(frame=frame,track=int(track),label=category.category_name,confidence=score,landmark_match_median_px=float(cost[detection,track])))
    evidence=dict(updated_utc=utc(),frames=cfg['frames'],votes=votes.tolist(),matched_observations=observations.tolist(),
                  original_hand_measurements_sha256=digest(source),source_measurements_modified=False,
                  ground_truth_used=False,observations=rows)
    # Preserve the diagnostic even if admission rejects the hand identities.
    save(rec/'global-hand-identity-observations.json',evidence)
    try:permutation,confidence,selection=decide_observations(votes,observations,rows)
    except RuntimeError as error:
        save(report,dict(**evidence,status='failed',reason=str(error),anatomical_assignment_verified=False))
        raise
    for name in ['uv','relative_xyz','observed','score']:
        values[name]=values[name][:,permutation]
    np.savez_compressed(rec/'hand_measurements_anatomical.npz',**values)
    save(report,dict(updated_utc=utc(),frames=cfg['frames'],status='passed',method='Whole-trajectory confidence-weighted handedness votes on continuous Hungarian landmark tracks',
        observation_selection=selection,
        track_for_anatomical_left_right=permutation,votes=votes.tolist(),matched_observations=observations.tolist(),confidence=confidence,
        original_hand_measurements_sha256=digest(source),source_measurements_modified=False,ground_truth_used=False,
        label_convention='HandLandmarker Tasks labels on upright egocentric frames; directly verified on public 26 open palms, not legacy selfie-label assumptions',observations=rows))
    print('Global hand identity',permutation,'confidence',confidence,'observations',observations.tolist(),flush=True)
