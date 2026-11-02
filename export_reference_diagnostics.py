"""Describe the actual frozen source trajectory for both frontend generations."""
import json
from pathlib import Path
import numpy as np
from scipy.spatial.transform import Rotation
from improvement_common import save,digest


def ensure(root):
    root=Path(root);rec=root/'reconstruction';cfg=json.loads((root/'episode-config.json').read_text())
    path=rec/'scene_motion_physical.npz'
    with np.load(path) as data:
        cameras=data['camera_to_world'];objects=data['object_to_world']
    assert len(cameras)==len(objects)==cfg['frames'] and np.isfinite(cameras).all() and np.isfinite(objects).all()
    def steps(poses):
        translation=np.linalg.norm(np.diff(poses[:,:3,3],axis=0),axis=1)
        rotations=poses[:-1,:3,:3].transpose(0,2,1)@poses[1:,:3,:3]
        angular=Rotation.from_matrix(rotations).magnitude()
        return dict(translation_step_m_p50_p95_max=np.percentile(translation,[50,95,100]).tolist(),
                    rotation_step_rad_p50_p95_max=np.percentile(angular,[50,95,100]).tolist())
    common=dict(method='Descriptive continuity statistics from the exact frozen source trajectory; not a quality-pass certificate',
                trajectory_sha256=digest(path),frames=cfg['frames'],ground_truth_used=False)
    if not (rec/'camera-stability.json').exists():save(rec/'camera-stability.json',dict(common,**steps(cameras)))
    if not (rec/'pose-continuity.json').exists():
        save(rec/'pose-continuity.json',dict(common,objects={name:steps(objects[:,body]) for body,name in enumerate(cfg['objects'])}))
