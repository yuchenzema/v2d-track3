"""Inspect every recovered object/camera frame for discontinuities before policy training."""
from pathlib import Path
import json,numpy as np
from scipy.spatial.transform import Rotation
BASE=Path('/home/yu_eli/v2d-track3')
def inspect(r):
 c=json.loads((r/'episode-config.json').read_text());p=r/'reconstruction/scene_motion_physical.npz'
 if not p.exists():return
 d=np.load(p);rows={}
 for b,name in enumerate(c['objects']):
  M=d['object_to_world'][:,b];R=Rotation.from_matrix(M[:,:3,:3]);angles=(R[:-1].inv()*R[1:]).magnitude();steps=np.linalg.norm(np.diff(M[:,:3,3],axis=0),axis=1);rows[name]={'maximum_translation_step_m':float(steps.max()),'maximum_rotation_step_rad':float(angles.max()),'rotation_steps_over_0_8_rad':np.flatnonzero(angles>.8).tolist(),'translation_steps_over_0_06_m':np.flatnonzero(steps>.06).tolist()}
 out={'method':__doc__,'frames':c['frames'],'all_arrays_finite':all(np.isfinite(d[k]).all() for k in d.files),'maximum_camera_step_m':float(np.linalg.norm(np.diff(d['camera_to_world'][:,:3,3],axis=0),axis=1).max()),'objects':rows,'flags_are_for_review_not_ground_truth':True};(r/'evaluation/reconstruction-motion-diagnostics.json').write_text(json.dumps(out,indent=2));print(c['episode'],json.dumps(rows),flush=True)
if __name__=='__main__':
 for r in sorted((BASE/'runs').glob('episode_*_newton_batch_20261015')):inspect(r)
