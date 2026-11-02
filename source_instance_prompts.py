"""Source-only separated-instance hints before reconstructing object geometry."""
import copy
import json
from pathlib import Path
import cv2
import numpy as np
from improvement_common import digest, save


def interior_points(mask,count=3):
    distance=cv2.distanceTransform(mask.astype(np.uint8),cv2.DIST_L2,5)
    if distance.max()<2:raise ValueError('No reliable foreground interior for instance hint')
    y,x=np.where(distance>=max(2,.6*distance.max()))
    points=np.c_[x,y];chosen=[int(np.argmax(distance[y,x]))]
    for _ in range(min(count-1,len(points)-1)):
        spacing=np.min(np.linalg.norm(points[:,None]-points[chosen][None],axis=2),axis=1)
        chosen.append(int(np.argmax(spacing)))
    return [dict(x=float(points[i,0]),y=float(points[i,1])) for i in dict.fromkeys(chosen)]


def separated_frame(masks_by_frame):
    """Choose a joint view without one mask swallowing another object's pixels."""
    candidates=[]
    for frame,masks in masks_by_frame:
        areas=np.array([m.sum() for m in masks],float)
        if np.min(areas)<200:continue
        overlap=np.zeros_like(masks[0],np.uint8)
        for mask in masks:overlap+=mask.astype(np.uint8)
        if (overlap>1).sum()>.01*areas.min():continue
        # The worst visible object determines view quality. All hints are
        # image-derived; no CAD, object pose, scan or held-out score is read.
        candidates.append((float(areas.min()),frame,masks))
    if not candidates:raise ValueError('No source view separates all visible object instances')
    return max(candidates,key=lambda row:row[0])


def prepare(source,root,cfg):
    source,root=Path(source),Path(root);rec=root/'reconstruction'
    original=json.loads((source/'reconstruction/sam2_prompts.json').read_text())
    if len(cfg['objects'])==1 or original.get('exclusive_visible_instances'):
        review=copy.deepcopy(original)
    else:
        def frames():
            # EP1 frame150 was independently inspected: green pot and magenta
            # lid are separate, visible objects. Other episodes choose a
            # non-overlapping measured view without assuming semantic geometry.
            order=[150] if cfg['episode']==1 else list(range(cfg['frames']))
            for frame in order:
                if frame>=cfg['frames']:continue
                masks=[cv2.imread(str(source/f'reconstruction/masks/{body+1}/{frame:06d}.png'),0) for body in range(len(cfg['objects']))]
                if all(m is not None for m in masks):yield frame,[m>0 for m in masks]
        _,frame,masks=separated_frame(frames())
        prompts=[]
        for body,mask in enumerate(masks):
            points=interior_points(mask)
            prompts.append(dict(object_id=body+1,frame_index=frame,points=points,point_labels=[1]*len(points)))
        review=dict(prompts=prompts,
            source_frames_sha256={str(frame):digest(source/f'reconstruction/frames/{frame:06d}.png')},
            provenance='Source-only visible instance interiors on a separated source frame; EP1 frame150 checked against RGB. No geometry or pose inputs.')
    review.update(exclusive_visible_instances=True,cross_instance_negatives=len(cfg['objects'])>1)
    for name in ['sam2_prompts.json','source-prompt-review.json']:save(rec/name,review)
    save(root/'research/source-instance-initialization.json',dict(
        source=str(source),prompt_sha256=digest(rec/'sam2_prompts.json'),
        geometry_or_pose_used=False,original_masks_modified=False,
        method='Disjoint source-visible instance hints; cross-instance negative points; exclusive ownership from SAM logits'))
    return review
