"""Measured source-image correspondences as rigid object-pose proposals."""
import cv2
import numpy as np


def penalty(pose,evidence,intr,independent_markers=False):
    # Fully checked current-frame fiducials must be able to correct a wrong
    # previous pose. Image tracks inherit that previous CAD correspondence.
    if evidence is None or independent_markers:return 0.
    points=evidence['object_points']@pose[:3,:3].T+pose[:3,3]
    if np.any(points[:,2]<=.05):return 100.
    uv=points[:,:2]/points[:,2:]*[intr['fx'],intr['fy']]+[intr['cx'],intr['cy']]
    residual=float(np.median(np.linalg.norm(uv-evidence['image_points'],axis=1)))
    return .06*float(np.hypot(1.,residual/3.)-1.)


def track(previous_image,image,renderer,previous_pose,intr,previous_mask,current_mask,
          previous_occlusion=None,current_occlusion=None,initial_pose=None):
    """Track visible source pixels and lift them using the previous CAD surface.

    Missing/occluded/ambiguous tracks return no proposal. No reference poses,
    object-marker identities or hidden image coordinates are supplied.
    """
    h,w=previous_mask.shape
    depth=renderer.depth(previous_pose)
    visible=cv2.resize((depth>.05).astype('uint8'),(w,h),interpolation=cv2.INTER_NEAREST)>0
    visible &= previous_mask.astype(bool)
    if previous_occlusion is not None:visible &= ~previous_occlusion.astype(bool)
    visible=cv2.erode(visible.astype('uint8'),np.ones((3,3),'uint8'))
    before=cv2.cvtColor(previous_image,cv2.COLOR_BGR2GRAY)
    after=cv2.cvtColor(image,cv2.COLOR_BGR2GRAY)
    points=cv2.goodFeaturesToTrack(before,160,.01,4,mask=visible,blockSize=5)
    if points is None or len(points)<8:return None
    options=dict(winSize=(15,15),maxLevel=3,criteria=(cv2.TERM_CRITERIA_EPS|cv2.TERM_CRITERIA_COUNT,30,.01))
    moved,status,error=cv2.calcOpticalFlowPyrLK(before,after,points,None,**options)
    if moved is None:return None
    returned,back_status,_=cv2.calcOpticalFlowPyrLK(after,before,moved,None,**options)
    if returned is None:return None
    p=points[:,0].astype(float);q=moved[:,0].astype(float)
    fb=np.linalg.norm(returned[:,0]-p,axis=1)
    keep=(status[:,0]>0)&(back_status[:,0]>0)&np.isfinite(q).all(1)&(fb<1.)&(error[:,0]<30.)
    indexes=np.rint(q).astype(int)
    inside=(indexes[:,0]>=0)&(indexes[:,0]<w)&(indexes[:,1]>=0)&(indexes[:,1]<h)
    keep &= inside
    ids=np.flatnonzero(inside)
    keep[ids] &= current_mask[indexes[ids,1],indexes[ids,0]].astype(bool)
    if current_occlusion is not None:
        keep[ids] &= ~current_occlusion[indexes[ids,1],indexes[ids,0]].astype(bool)
    dh,dw=depth.shape
    sample=np.rint(p*[dw/w,dh/h]).astype(int)
    sample[:,0]=sample[:,0].clip(0,dw-1);sample[:,1]=sample[:,1].clip(0,dh-1)
    z=depth[sample[:,1],sample[:,0]]
    keep &= np.isfinite(z)&(z>.05)
    p,q,z,fb=p[keep],q[keep],z[keep],fb[keep]
    if len(p)<8 or min(np.ptp(p,axis=0))<10.:return None
    camera=np.c_[(p[:,0]-intr['cx'])/intr['fx']*z,(p[:,1]-intr['cy'])/intr['fy']*z,z]
    xyz=(camera-previous_pose[:3,3])@previous_pose[:3,:3]
    K=np.array([[intr['fx'],0.,intr['cx']],[0.,intr['fy'],intr['cy']],[0.,0.,1.]])
    initial=previous_pose if initial_pose is None else initial_pose
    rv=cv2.Rodrigues(initial[:3,:3])[0];tv=initial[:3,3].copy().reshape(3,1)
    try:
        ok,rv,tv,inliers=cv2.solvePnPRansac(xyz,q,K,None,rvec=rv,tvec=tv,useExtrinsicGuess=True,
            iterationsCount=150,reprojectionError=2.5,confidence=.999,flags=cv2.SOLVEPNP_ITERATIVE)
        if not ok or inliers is None or len(inliers)<8 or len(inliers)<.6*len(p):return None
        ids=inliers[:,0]
        rv,tv=cv2.solvePnPRefineLM(xyz[ids],q[ids],K,None,rv,tv)
    except cv2.error:return None
    pose=np.eye(4);pose[:3,:3]=cv2.Rodrigues(rv)[0];pose[:3,3]=tv[:,0]
    projected=cv2.projectPoints(xyz[ids],rv,tv,K,None)[0][:,0]
    residual=float(np.median(np.linalg.norm(projected-q[ids],axis=1)))
    if not np.isfinite(pose).all() or pose[2,3]<=.05 or residual>2.5:return None
    return dict(pose=pose,object_points=xyz[ids],image_points=q[ids],
                previous_image_points=p[ids],tracked_count=len(p),inlier_count=len(ids),
                median_reprojection_error_px=residual,median_forward_backward_error_px=float(np.median(fb[ids])))
