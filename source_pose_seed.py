"""Carry a reconstructed rigid pose between independently calibrated world frames."""
import numpy as np

def rebase_reconstructed_pose(new_camera,old_camera,old_object):
    transforms=[np.asarray(value,dtype=float) for value in [new_camera,old_camera,old_object]]
    for transform in transforms:
        if transform.shape!=(4,4) or not np.isfinite(transform).all():raise ValueError('Invalid rigid source-pose seed')
        if not np.allclose(transform[3],[0,0,0,1],atol=1e-6):raise ValueError('Source transform is not homogeneous')
        rotation=transform[:3,:3]
        if not np.allclose(rotation.T@rotation,np.eye(3),atol=1e-3) or not np.isclose(np.linalg.det(rotation),1.,atol=1e-3):
            raise ValueError('Pose seed must not rescale or reflect the object')
    return transforms[0]@np.linalg.solve(transforms[1],transforms[2])
