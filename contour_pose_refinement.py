"""Fast local rigid-pose proposals from visible CAD/source contour correspondences.

The caller rechecks every candidate against its unchanged multi-signal objective
and strict source thresholds. This does not certify a reconstruction or policy.
"""
import cv2
import numpy as np
from scipy.optimize import least_squares
from scipy.spatial import cKDTree
from scipy.spatial.transform import Rotation


def surface_center(renderer):
    """A geometric centre must not move when one surface is more tessellated."""
    center=getattr(renderer,'_proposal_surface_center',None)
    if center is None:
        triangles=renderer.vertices[renderer.faces]
        areas=np.linalg.norm(np.cross(triangles[:,1]-triangles[:,0],triangles[:,2]-triangles[:,0]),axis=1)
        center=(np.average(triangles.mean(axis=1),axis=0,weights=areas) if areas.sum()>1e-12 else renderer.vertices.mean(axis=0))
        renderer._proposal_surface_center=center
    return center


def face_flip_proposals(renderer,seed):
    """Proper rotations around the mesh centre for two-sided thin objects."""
    center=surface_center(renderer)
    _,axes=np.linalg.eigh(np.cov(renderer.vertices.T));result=[]
    for axis in axes[:,1:].T:
        M=seed.copy();M[:3,:3]=seed[:3,:3]@Rotation.from_rotvec(axis*np.pi).as_matrix()
        M[:3,3]=seed[:3,3]+seed[:3,:3]@center-M[:3,:3]@center
        result.append(M)
    return result


def moving_orientation_proposals(renderer,seed,camera_to_world,previous,earlier=None):
    """Offer continuous orientations at the current measured mesh centroid.

    Reusing a whole previous pose prevents recovery when a thin object moves
    quickly across the image. These proposals retain the current translation
    evidence and still require the caller's silhouette/depth/marker checks.
    """
    center=surface_center(renderer)
    centroid=seed[:3,:3]@center+seed[:3,3]
    camera_rotation=camera_to_world[:3,:3].T
    rotations=[previous[:3,:3]]
    if earlier is not None:
        velocity=Rotation.from_matrix(previous[:3,:3]@earlier[:3,:3].T).magnitude()
        if velocity<.5:rotations.append(previous[:3,:3]@earlier[:3,:3].T@previous[:3,:3])
    # For an apparent face inversion, use the reflection axis closest to the
    # preceding orientation, rather than arbitrary PCA axes in a circular lid.
    _,axes=np.linalg.eigh(np.cov(renderer.vertices.T));normal=axes[:,0];plane=axes[:,1:]
    prior_local=seed[:3,:3].T@camera_rotation@previous[:3,:3]
    normal_agreement=float(normal@prior_local@normal)
    if normal_agreement<-.3:
        projected=plane.T@prior_local@plane
        _,directions=np.linalg.eigh((projected+projected.T)/2)
        axis=plane@directions[:,-1]
        flipped=seed[:3,:3]@Rotation.from_rotvec(axis*np.pi).as_matrix()
        rotations.append(camera_to_world[:3,:3]@flipped)
    result=[]
    for world_rotation in rotations:
        pose=seed.copy();pose[:3,:3]=camera_rotation@world_rotation
        pose[:3,3]=centroid-pose[:3,:3]@center;result.append(pose)
    return result


def temporal_pose_cost(world,previous,earlier):
    acceleration=world[:3,3]-2*previous[:3,3]+earlier[:3,3]
    predicted=previous[:3,:3]@earlier[:3,:3].T@previous[:3,:3]
    angular=Rotation.from_matrix(predicted.T@world[:3,:3]).magnitude()
    # A bounded ambiguity penalty can be outweighed by strong source evidence.
    return .001*float(np.sum((acceleration/.025)**2))+.001*min(float(angular/.15),10.)**2


def propose(renderer,seed,target,valid,intrinsics,measured=None,rounds=3):
    pose=seed.copy();height,width=target.shape
    image_width=float(intrinsics.get('width',2*width));image_height=float(intrinsics.get('height',2*height))
    fx=intrinsics['fx']*width/image_width;fy=intrinsics['fy']*height/image_height;cx=intrinsics['cx']*width/image_width;cy=intrinsics['cy']*height/image_height
    interior=cv2.erode(valid.astype(np.uint8),np.ones((3,3),np.uint8))>0
    edge=target&~cv2.erode(target.astype(np.uint8),np.ones((3,3),np.uint8)).astype(bool)&interior
    ty,tx=np.where(edge)
    if len(tx)<30:return None
    target_points=np.c_[tx,ty].astype(float)
    if len(target_points)>500:target_points=target_points[np.linspace(0,len(target_points)-1,500).round().astype(int)]
    tree=cKDTree(target_points)
    for _ in range(rounds):
        mask=renderer.render(pose);surface=renderer.depth(pose)
        contour=mask&~cv2.erode(mask.astype(np.uint8),np.ones((3,3),np.uint8)).astype(bool)&interior&(surface>.05)
        y,x=np.where(contour)
        if len(x)<30:return None
        if len(x)>500:
            ids=np.linspace(0,len(x)-1,500).round().astype(int);x=x[ids];y=y[ids]
        pixels=np.c_[x,y].astype(float);z=surface[y,x]
        camera_points=np.c_[(x-cx)/fx*z,(y-cy)/fy*z,z]
        object_points=(camera_points-pose[:3,3])@pose[:3,:3]
        distance,ids=tree.query(pixels);keep=distance<35.
        reverse_distance,reverse_ids=cKDTree(pixels).query(target_points);reverse_keep=reverse_distance<35.
        if keep.sum()<20:return None
        xyz=np.r_[object_points[keep],object_points[reverse_ids[reverse_keep]]]
        observed=np.r_[target_points[ids[keep]],target_points[reverse_keep]]
        depth_sample=None
        if measured is not None:
            u,v=np.rint(observed).astype(int).T;depth_sample=measured[v,u];has_depth=np.isfinite(depth_sample)&(depth_sample>.05)
        base=pose.copy()
        def candidate(delta):
            M=base.copy();M[:3,3]+=delta[:3];M[:3,:3]=Rotation.from_rotvec(delta[3:]).as_matrix()@base[:3,:3];return M
        def residual(delta):
            M=candidate(delta);points=xyz@M[:3,:3].T+M[:3,3]
            projection=points[:,:2]/np.maximum(points[:,2:],.05)*[fx,fy]+[cx,cy]
            values=[(projection-observed).ravel(),.1*delta[:3]/.03,.1*delta[3:]/.25]
            if depth_sample is not None and has_depth.any():
                values.append(.35*(points[has_depth,2]-depth_sample[has_depth])*np.sqrt(fx*fy)/np.maximum(points[has_depth,2],.05))
            return np.concatenate(values)
        fitted=least_squares(residual,np.zeros(6),bounds=([-.03]*3+[-.25]*3,[.03]*3+[.25]*3),loss='soft_l1',f_scale=2.,max_nfev=35)
        if not fitted.success or not np.isfinite(fitted.x).all():return None
        pose=candidate(fitted.x)
    return pose
