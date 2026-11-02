"""Conservative forearm evidence from source hands and stereo-aligned depth.

No CAD pose, rendered silhouette or target pose is used. Existing source object
foreground is protected. Missing depth never supplies an occluding pixel.
"""
import cv2
import numpy as np


def measured_forearms(hands,frame,depth,object_foreground):
    height,width=depth.shape;scale=np.array([width/1024.,height/768.])
    protected=np.asarray(object_foreground,dtype=bool)
    result=np.zeros(depth.shape,np.uint8)
    valid=np.isfinite(depth)&(depth>.05)&(depth<3.)
    for side in range(2):
        if not hands['observed'][frame,side]:continue
        points=np.asarray(hands['uv'][frame,side],float)*scale
        if not np.isfinite(points).all():continue
        wrist=points[0];knuckles=points[[5,9,13,17]].mean(axis=0)
        direction=wrist-knuckles;length=np.linalg.norm(direction)
        if length<3.:continue
        direction/=length;across=np.array([-direction[1],direction[0]])
        palm_width=np.linalg.norm(points[5]-points[17]);half_width=float(np.clip(palm_width*1.2,8.,70.))
        near=wrist-direction*min(length,12.);far=wrist+direction*np.hypot(width,height)
        corridor=np.zeros(depth.shape,np.uint8)
        polygon=np.rint([near-across*half_width,near+across*half_width,far+across*half_width*1.4,far-across*half_width*1.4]).astype(np.int32)
        cv2.fillConvexPoly(corridor,polygon,1)
        palm=np.zeros(depth.shape,np.uint8)
        cv2.fillConvexPoly(palm,cv2.convexHull(np.rint(points[[0,1,5,9,13,17]]).astype(np.int32)),1)
        samples=depth[(palm>0)&valid&~protected]
        if len(samples)<12:continue
        # Toward the camera, a visible forearm is no deeper than its hand.
        # Reject the table/background instead of growing an image-space strip.
        maximum=float(np.quantile(samples,.75))+.025
        candidate=(corridor>0)&valid&(depth<=maximum)&~protected
        seeds=cv2.dilate(palm,np.ones((5,5),np.uint8))>0
        connected=cv2.morphologyEx(candidate.astype(np.uint8),cv2.MORPH_CLOSE,np.ones((7,7),np.uint8))
        _,labels=cv2.connectedComponents(connected,connectivity=8)
        touched=np.unique(labels[seeds&candidate]);touched=touched[touched!=0]
        if len(touched):result[np.isin(labels,touched)&candidate]=1
    assert not np.any(result.astype(bool)&protected)
    assert not np.any(result.astype(bool)&~valid)
    return result


def attach(root,hands,count,object_count):
    """Attach small source-only masks; preserve original depth/images/object masks."""
    from pathlib import Path
    from improvement_common import digest,save,utc
    root=Path(root);rec=root/'reconstruction';evidence=[];masks=[]
    folder=rec/'source-forearm-occlusion';folder.mkdir(exist_ok=True)
    for frame in range(count):
        depth_path=rec/f'depth/{frame:06d}.png'
        if not depth_path.exists():depth_path=rec/f'depth_stereo/{frame:06d}.png'
        depth=cv2.imread(str(depth_path),cv2.IMREAD_UNCHANGED) if depth_path.exists() else None
        output=np.zeros((384,512),np.uint8);record=dict(frame=frame,occluding_pixels=0,depth_available=False)
        if depth is not None:
            values=np.where(depth>0,65535/np.maximum(depth.astype(float),1)-1,0)
            values=cv2.resize(values,(512,384),interpolation=cv2.INTER_NEAREST)
            paths=[rec/f'masks/{body+1}/{frame:06d}.png' for body in range(object_count)]
            foreground=np.zeros((384,512),bool)
            for path in paths:
                source=cv2.imread(str(path),0);assert source is not None
                foreground|=cv2.resize(source,(512,384),interpolation=cv2.INTER_NEAREST)>0
            output=measured_forearms(hands,frame,values,foreground)
            record.update(depth_available=True,depth_path=str(depth_path.relative_to(rec)),depth_sha256=digest(depth_path),
                          object_masks_sha256={str(p.relative_to(rec)):digest(p) for p in paths},occluding_pixels=int(output.sum()))
        destination=folder/f'{frame:06d}.png';assert cv2.imwrite(str(destination),output*255)
        record['derived_mask_sha256']=digest(destination);evidence.append(record);masks.append(output)
    augmented={key:hands[key] for key in hands}
    augmented['forearm_masks']=masks
    save(root/'research/source-forearm-occlusion.json',dict(updated_utc=utc(),
        algorithm_sha256=digest(Path(__file__)),hand_measurements_sha256=digest(rec/'hand_measurements.npz'),
        method='Hand-connected foreground in an anatomically directed corridor, from independent source metric depth; all source object foreground protected',
        depth_is_estimated_not_ground_truth=True,cad_pose_used=False,source_images_modified=False,
        source_depth_modified=False,object_masks_modified=False,missing_depth_excluded=True,
        frames=evidence,does_not_grant_episode_completion=True))
    return augmented
