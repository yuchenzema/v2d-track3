"""Source-image hint checks for every object, and white-mask wood leakage.

This check catches connected wooden tabletop leakage. It does not certify a
mask's geometry, hidden surface, object identity, or pose accuracy.
"""
from pathlib import Path
import json
import cv2
import numpy as np
from improvement_common import save,utc,digest

WHITE_OBJECTS={'white_pot','white_pot_lid','plastic_dish_rack'}
VERSION=3


def merge_repair_prompt(prompts,body,frame,repair):
    """Keep reviewed foreground semantics, including disconnected handles."""
    prior=next((p for p in prompts['prompts'] if p['object_id']==body and p['frame_index']==frame and p.get('points')),None)
    if prior is None:
        prompts['prompts']=[p for p in prompts['prompts'] if not (p['object_id']==body and p['frame_index']==frame)]
        prompts['prompts'].append(dict(object_id=body,frame_index=frame,**repair));return
    labels=prior.setdefault('point_labels',[1]*len(prior['points']))
    positives=[np.array([p['x'],p['y']]) for p,label in zip(prior['points'],labels) if label==1]
    for point,label in zip(repair['points'],repair['point_labels']):
        # Existing reviewed positives already define the complete object. Only
        # add new background exclusions; never replace them with one body blob.
        if label!=0:continue
        value=np.array([point['x'],point['y']])
        if any(np.linalg.norm(value-old)<12 for old in positives):continue
        if any(np.linalg.norm(value-np.array([p['x'],p['y']]))<2 for p in prior['points']):continue
        prior['points'].append(point);labels.append(label)


def contamination(image,mask,excluded=None):
    hsv=cv2.cvtColor(image,cv2.COLOR_BGR2HSV);visible=mask>0
    if excluded is not None:visible &= ~excluded
    wood=(hsv[:,:,0]>4)&(hsv[:,:,0]<28)&(hsv[:,:,1]>85)&(hsv[:,:,2]>65)&visible
    count=int(wood.sum());area=int(visible.sum())
    return dict(visible_pixels=area,wood_pixels=count,wood_fraction=count/max(1,area),
                contaminated=count>2000 and count/max(1,area)>.25)


def repair_points(image,mask,other_masks):
    hsv=cv2.cvtColor(image,cv2.COLOR_BGR2HSV);inside=mask>0
    white=inside&(hsv[:,:,1]<70)&(hsv[:,:,2]>100)
    for other in other_masks:white &= other==0
    count,labels,stats,_=cv2.connectedComponentsWithStats(white.astype(np.uint8))
    if count<2:raise RuntimeError('No independent white-object component for a repair prompt')
    label=1+np.argmax(stats[1:,cv2.CC_STAT_AREA]);component=(labels==label).astype(np.uint8)
    if component.sum()<100:raise RuntimeError('White-object repair evidence is too small')
    distance=cv2.distanceTransform(component,cv2.DIST_L2,5);y,x=np.unravel_index(np.argmax(distance),distance.shape)
    points=[dict(x=float(x),y=float(y))];point_labels=[1]
    wood=inside&(hsv[:,:,0]>4)&(hsv[:,:,0]<28)&(hsv[:,:,1]>85)&(hsv[:,:,2]>65)
    for negative in [wood,*[m>0 for m in other_masks]]:
        if negative.sum()<50:continue
        distance=cv2.distanceTransform(negative.astype(np.uint8),cv2.DIST_L2,5);y,x=np.unravel_index(np.argmax(distance),distance.shape)
        points.append(dict(x=float(x),y=float(y)));point_labels.append(0)
    if len(points)<2:raise RuntimeError('No measured background exclusion for source-mask repair')
    return dict(points=points,point_labels=point_labels)


def audit(root):
    root=Path(root);rec=root/'reconstruction';cfg=json.loads((root/'episode-config.json').read_text());hands=np.load(rec/'hand_measurements.npz');failures=[]
    prompts=[]
    for prompt_file in [rec/'source-prompt-review.json',rec/'sam2_prompts.json']:
        if prompt_file.exists():prompts.extend(json.loads(prompt_file.read_text()).get('prompts',[]))
    for frame in range(cfg['frames']):
        image=cv2.imread(str(rec/f'frames/{frame:06d}.png'));excluded=np.zeros(image.shape[:2],np.uint8)
        for side in range(2):
            if hands['observed'][frame,side]:cv2.fillConvexPoly(excluded,cv2.convexHull(np.rint(hands['uv'][frame,side]).astype(np.int32)),1)
        excluded=cv2.dilate(excluded,np.ones((17,17),np.uint8))>0
        for body,name in enumerate(cfg['objects'],1):
            mask=cv2.imread(str(rec/f'masks/{body}/{frame:06d}.png'),0)
            if mask is None:raise RuntimeError(f'Missing original source mask {name} frame {frame}')
            row=(contamination(image,mask,excluded) if name in WHITE_OBJECTS else
                 dict(visible_pixels=int(((mask>0)&~excluded).sum()),wood_pixels=0,wood_fraction=0.,contaminated=False))
            missing=[];included=[]
            for prompt in prompts:
                if prompt['object_id']!=body or prompt['frame_index']!=frame:continue
                for point,label in zip(prompt.get('points',[]),prompt.get('point_labels',[])):
                    x,y=np.rint([point['x'],point['y']]).astype(int)
                    if not (0<=y<mask.shape[0] and 0<=x<mask.shape[1]):continue
                    if label==1 and not excluded[y,x] and not mask[max(0,y-2):y+3,max(0,x-2):x+3].any() and point not in missing:missing.append(point)
                    if label==0 and mask[y,x] and point not in included:included.append(point)
            if row['contaminated'] or missing or included:failures.append(dict(frame=frame,body=body,object=name,
                missing_reviewed_foreground_points=missing,included_reviewed_background_points=included,**row))
    return dict(version=VERSION,updated_utc=utc(),frames_checked=cfg['frames'],failure_count=len(failures),failures=failures,
        scope='Foreground and background hints checked for every object; wooden-table contamination checked for white objects only. Unhinted segmentation errors remain unverified.',ground_truth_used=False)


def inspect_and_repair(root,rerun):
    root=Path(root);rec=root/'reconstruction';cfg=json.loads((root/'episode-config.json').read_text());history=[]
    for attempt in range(3):
        report=audit(root);history.append(dict(attempt=attempt,**report));save(root/'research/source-mask-quality.json',dict(attempts=history,latest=report))
        if not report['failures']:return
        if attempt==2:raise RuntimeError(f"Source mask contamination or foreground/background-hint failure persists in {report['failure_count']} object-frames")
        prompts=json.loads((rec/'sam2_prompts.json').read_text())
        for body in sorted({row['body'] for row in report['failures']}):
            # A hole or other reviewed hint must be honoured by the full prompt;
            # a white-component heuristic is invalid for coloured objects.
            rows=sorted((row for row in report['failures'] if row['body']==body and row['contaminated']),key=lambda row:row['wood_pixels'],reverse=True)
            reviewed_points=[p for p in prompts['prompts'] if p['object_id']==body and p.get('points')]
            if reviewed_points:
                # Repair the closest complete conditioning frame rather than
                # inventing a new incomplete object definition on another frame.
                available=sorted({p['frame_index'] for p in reviewed_points})
                rows=[dict(row,frame=min(available,key=lambda f:abs(f-row['frame']))) for row in rows]
            chosen=[]
            for row in rows:
                if all(abs(row['frame']-old['frame'])>30 for old in chosen):chosen.append(row)
                if len(chosen)==3:break
            for row in chosen:
                frame=row['frame'];image=cv2.imread(str(rec/f'frames/{frame:06d}.png'));mask=cv2.imread(str(rec/f'masks/{body}/{frame:06d}.png'),0)
                other=[cv2.imread(str(rec/f'masks/{b}/{frame:06d}.png'),0) for b in range(1,len(cfg['objects'])+1) if b!=body]
                merge_repair_prompt(prompts,body,frame,repair_points(image,mask,other))
        prompts.update(provenance='Source-only reviewed foreground/background preservation and white-object wooden-table exclusion where needed; no pose GT used')
        save(rec/f'sam2_prompts.before_auto_repair_{attempt}.json',json.loads((rec/'sam2_prompts.json').read_text()))
        save(rec/'sam2_prompts.json',prompts);rerun()
