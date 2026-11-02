"""Metric camera motion from background SIFT correspondences and stereo PnP.

Task-object masks and measured hands are excluded. No object poses or Vicon
measurements are used. Failed motion estimates stop this stage explicitly.
"""
from pathlib import Path
import json
import cv2
import numpy as np
from scipy.spatial.transform import Rotation
from improvement_common import save,utc


def estimate_relative(points,pixels,K):
    points=np.asarray(points,np.float64);pixels=np.asarray(pixels,np.float64)
    if len(points)<12:raise RuntimeError(f'Only {len(points)} metric background matches')
    cv2.setRNGSeed(731)
    ok,rvec,tvec,inliers=cv2.solvePnPRansac(points,pixels,K,None,
        iterationsCount=500,reprojectionError=3.,confidence=.999,flags=cv2.SOLVEPNP_EPNP)
    if not ok or inliers is None or len(inliers)<12:
        raise RuntimeError('Metric background PnP has fewer than 12 verified inliers')
    ids=inliers[:,0]
    rvec,tvec=cv2.solvePnPRefineLM(points[ids],pixels[ids],K,None,rvec,tvec)
    error=np.linalg.norm(cv2.projectPoints(points[ids],rvec,tvec,K,None)[0][:,0]-pixels[ids],axis=1)
    if np.median(error)>1.5:raise RuntimeError('Metric camera PnP reprojection error exceeds 1.5 pixels')
    M=np.eye(4);M[:3,:3]=cv2.Rodrigues(rvec)[0];M[:3,3]=tvec[:,0]
    return M,dict(matches=len(points),inliers=len(ids),median_reprojection_error_px=float(np.median(error)))


def run(root):
    root=Path(root);rec=root/'reconstruction';cfg=json.loads((root/'episode-config.json').read_text())
    output=rec/'sift-pnp-odometry.json'
    if output.exists():return
    intr=json.loads((rec/'camera_intrinsics.json').read_text())
    K=np.array([[intr['fx'],0,intr['cx']],[0,intr['fy'],intr['cy']],[0,0,1.]])
    hands=np.load(rec/'hand_measurements.npz');sift=cv2.SIFT_create(nfeatures=6000);matcher=cv2.BFMatcher()
    poses=[];reports=[];previous=None
    folder=rec/'slam/poses';folder.mkdir(parents=True,exist_ok=True)
    for frame in range(cfg['frames']):
        gray=cv2.imread(str(rec/f'frames/{frame:06d}.png'),0)
        allowed=(gray>5).astype(np.uint8)*255
        for body in range(1,len(cfg['objects'])+1):
            mask=cv2.imread(str(rec/f'masks/{body}/{frame:06d}.png'),0)
            allowed[cv2.dilate(mask,np.ones((17,17),np.uint8))>0]=0
        for side in range(2):
            if hands['observed'][frame,side]:
                mask=np.zeros_like(gray);cv2.fillConvexPoly(mask,cv2.convexHull(np.rint(hands['uv'][frame,side]).astype(np.int32)),255)
                allowed[cv2.dilate(mask,np.ones((25,25),np.uint8))>0]=0
        kp,desc=sift.detectAndCompute(gray,allowed)
        encoded=cv2.imread(str(rec/f'depth_foundation_stereo/{frame:06d}.png'),-1)
        depth=65535/np.maximum(encoded.astype(float),1)-1
        if previous is None:M=np.eye(4);row=dict(frame=0,inliers=0,method='world-frame gauge')
        else:
            old_kp,old_desc,old_depth,old_encoded=previous;points=[];pixels=[]
            if old_desc is None or desc is None:raise RuntimeError(f'Background SIFT descriptors missing at frame {frame}')
            for pair in matcher.knnMatch(old_desc,desc,k=2):
                if len(pair)!=2 or pair[0].distance>=.7*pair[1].distance:continue
                match=pair[0];u,v=old_kp[match.queryIdx].pt;x,y=np.rint([u,v]).astype(int)
                if not (0<=y<old_depth.shape[0] and 0<=x<old_depth.shape[1]):continue
                z=old_depth[y,x]
                if old_encoded[y,x]==0 or not .2<z<5.:continue
                points.append([(u-intr['cx'])/intr['fx']*z,(v-intr['cy'])/intr['fy']*z,z])
                pixels.append(kp[match.trainIdx].pt)
            relative,row=estimate_relative(points,pixels,K)
            if np.linalg.norm(relative[:3,3])>.2 or Rotation.from_matrix(relative[:3,:3]).magnitude()>.4:
                raise RuntimeError(f'Implausible per-frame metric camera jump at frame {frame}')
            M=poses[-1]@np.linalg.inv(relative);row['frame']=frame
        poses.append(M);reports.append(row)
        save(folder/f'{frame:06d}.json',dict(rotation=Rotation.from_matrix(M[:3,:3]).as_quat(scalar_first=True).tolist(),translation=M[:3,3].tolist()))
        previous=(kp,desc,depth,encoded)
        if frame%10==0:
            save(root/'research/camera-sift-progress.json',dict(updated_utc=utc(),episode=cfg['episode'],frame=frame+1,total_frames=cfg['frames'],phase='Metric background SIFT + PnP-RANSAC camera motion',policy_training_started=False))
            print('SIFT PnP',frame+1,'/',cfg['frames'],'inliers',row['inliers'],flush=True)
    np.savez_compressed(rec/'sift-pnp-odometry.npz',camera_to_world=np.array(poses))
    save(output,dict(updated_utc=utc(),frames=cfg['frames'],method='Background SIFT, metric FoundationStereo points, PnP-RANSAC and inlier-only LM',task_objects_and_hands_excluded=True,ground_truth_used=False,per_frame=reports))
