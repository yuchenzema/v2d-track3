"""Reinitialise translation from measured metric surfaces, without pose labels."""
import numpy as np


def translation_seed(renderer,pose,intrinsics,measured,target,visibility=None):
    valid=target&(measured>.05)&np.isfinite(measured)
    rendered=renderer.depth(pose);model=(rendered>.05)&np.isfinite(rendered)
    # Both centroids must describe the same observable image region. Comparing
    # a partly hand-occluded measurement with the complete CAD surface creates
    # translation even when the input pose already matches the source exactly.
    if visibility is not None:
        visible=np.asarray(visibility,dtype=bool)
        if visible.shape!=target.shape:raise ValueError('Visibility and metric target shapes differ')
        valid &= visible;model &= visible
    if valid.sum()<60 or model.sum()<60:return None
    def center(depth,mask):
        v,u=np.where(mask);z=depth[v,u]
        median=np.median(z);mad=np.median(np.abs(z-median));keep=np.abs(z-median)<max(.02,3*mad)
        v,u,z=v[keep],u[keep],z[keep]
        points=np.c_[(u-intrinsics['cx']/2)/(intrinsics['fx']/2)*z,(v-intrinsics['cy']/2)/(intrinsics['fy']/2)*z,z]
        return np.median(points,axis=0)
    delta=center(measured,valid)-center(rendered,model)
    if not np.isfinite(delta).all() or np.linalg.norm(delta)>.6:return None
    result=pose.copy();result[:3,3]+=delta
    return result if result[2,3]>.05 else None
