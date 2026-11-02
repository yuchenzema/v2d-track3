"""Keep inferred rigid geometry above the finite physical tabletop."""
import numpy as np

def constrain_pose(pose,vertices,center,radius,surface,settled=False):
    result=np.asarray(pose,dtype=float).copy()
    world=np.asarray(vertices)@result[:3,:3].T+result[:3,3]
    inside=np.linalg.norm(world[:,:2]-np.asarray(center)[:2],axis=1)<radius
    if inside.any():
        lift=float(surface-world[inside,2].min())
        result[2,3]+=lift if settled else max(0.,lift)
    return result
