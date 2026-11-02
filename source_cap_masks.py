"""Recover only observed grey cap interiors at source-reviewed image anchors.

This is source segmentation, independent of CAD poses and pose ground truth.
Missing image evidence never supplies foreground, and another instance's
existing foreground is never reassigned.
"""
import json
import os
from pathlib import Path
import shutil

import cv2
import numpy as np

from improvement_common import digest,save,utc
from source_marker_pose_proposals import load_reviews
from sphere_marker_reference import circle_candidates


def writable_mask_directory(root,directory):
    if os.access(directory,os.W_OK):return
    # SAM2 creates these output directories as container root. Change only
    # this episode's mask directory owner, without a GPU or recursive chown.
    relative=directory.relative_to(root)
    from batch_native import _docker
    _docker(['run','--rm','-v',f'{root}:/episode','--entrypoint','/bin/chown','flash_chord',
             f'{os.getuid()}:{os.getgid()}','/episode/'+str(relative)],
            root/'logs/fix-source-cap-mask-directory-owner.log')


def refine(root,review_directory=None):
    root=Path(root);rec=root/'reconstruction'
    cfg=json.loads((root/'episode-config.json').read_text())
    if not cfg.get('validated_video_pipeline'):return None
    reviews,binding=load_reviews(cfg,rec,review_directory)
    if not reviews:return None
    frames=sorted({row['frame'] for row in reviews});records=[];mask_hashes={};image_hashes={}
    for frame in frames:
        image_path=rec/f"frames/{frame:06d}.{cfg.get('frame_cache_format','png')}"
        image=cv2.imread(str(image_path));assert image is not None
        image_hashes[str(frame)]=digest(image_path)
        hsv=cv2.cvtColor(image,cv2.COLOR_BGR2HSV);yy,xx=np.indices(image.shape[:2])
        paths=[rec/f'masks/{i+1}/{frame:06d}.png' for i in range(len(cfg['objects']))]
        values=[cv2.imread(str(p),0) for p in paths];assert all(x is not None and x.shape==image.shape[:2] for x in values)
        masks=[x>0 for x in values];claims=[np.zeros(image.shape[:2],bool) for _ in masks]
        for review in [row for row in reviews if row['frame']==frame]:
            body=cfg['objects'].index(review['object'])
            occupied=np.zeros(image.shape[:2],bool)
            for i,mask in enumerate(masks):
                if i!=body:occupied|=mask
            for point in review['points']:
                candidates=[c for c in circle_candidates(image,np.array(point['xy'],float),point['radius_px'])
                    if c['cost']<4.5 and np.linalg.norm(np.array(c['xy'])-point['xy'])<=max(3.,.75*point['radius_px'])]
                if not candidates:
                    records.append(dict(frame=frame,object=review['object'],source_point=point,status='no_measured_cap'));continue
                measured=min(candidates,key=lambda c:c['cost']);x,y=measured['xy']
                # Conservative interior, rather than a projected or invented
                # full sphere silhouette. Every included pixel is measured grey.
                disk=(xx-x)**2+(yy-y)**2<(.8*measured['radius_px'])**2
                grey=(hsv[:,:,1]<85)&(hsv[:,:,2]>32)&(hsv[:,:,2]<235)
                measured_pixels=disk&grey
                claims[body]|=measured_pixels&~occupied
                records.append(dict(frame=frame,object=review['object'],source_point=point,status='measured_cap',
                    measurement=measured,measured_interior_pixels=int(measured_pixels.sum()),
                    existing_other_instance_pixels_skipped=int((measured_pixels&occupied).sum())))
        conflict=np.sum(claims,axis=0)>1
        for i,path in enumerate(paths):
            added=claims[i]&~conflict&~masks[i]
            if added.any():
                before=digest(path);backup=rec/f'masks-before-source-caps/{before}/{i+1}/{frame:06d}.png'
                backup.parent.mkdir(parents=True,exist_ok=True)
                if not backup.exists():shutil.copy2(path,backup)
                result=masks[i]|added;temporary=path.with_name('.'+path.stem+'.source-caps.png')
                writable_mask_directory(root,path.parent)
                assert cv2.imwrite(str(temporary),result.astype('uint8')*255)
                temporary.replace(path)
            mask_hashes[str(path.relative_to(rec))]=digest(path)
        records.append(dict(frame=frame,ambiguous_claim_pixels_skipped=int(conflict.sum()),
            source_foreground_pixels_added=sum(int((claims[i]&~conflict&~mask).sum()) for i,mask in enumerate(masks))))
    report=dict(updated_utc=utc(),method='Conservative measured grey-cap interiors at image-reviewed source points',
        input_prompts_sha256=digest(rec/'sam2_prompts.json'),marker_review=binding,
        source_images_sha256=image_hashes,masks_sha256=mask_hashes,measurements=records,
        ground_truth_used=False,cad_pose_used=False,source_images_modified=False,
        existing_foreground_removed=False,existing_instance_labels_reassigned=False,
        full_mask_geometry_verified=False,strict_episode_completed=False)
    save(root/'research/source-cap-mask-refinement.json',report)
    return report


def verify(root,prompts_sha256,review_directory=None):
    root=Path(root);path=root/'research/source-cap-mask-refinement.json'
    if not path.exists():return None
    report=json.loads(path.read_text());rec=root/'reconstruction'
    if report.get('input_prompts_sha256')!=prompts_sha256:
        raise RuntimeError('Source cap masks belong to different source prompts')
    cfg=json.loads((root/'episode-config.json').read_text());_,binding=load_reviews(cfg,rec,review_directory)
    if report.get('marker_review')!=binding:
        raise RuntimeError('Source cap masks belong to different reviewed image caps')
    for frame,expected in report['source_images_sha256'].items():
        if digest(rec/f"frames/{int(frame):06d}.{cfg.get('frame_cache_format','png')}")!=expected:
            raise RuntimeError('Source cap image evidence changed')
    for name,expected in report['masks_sha256'].items():
        if digest(rec/name)!=expected:raise RuntimeError('Source cap mask evidence changed')
    return dict(path=str(path),sha256=digest(path))
