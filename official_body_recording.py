"""Read-only capture of the very same official single-world policy rollout."""
import os
from pathlib import Path
import numpy as np


def install():
    target=os.environ.get('V2D_OFFICIAL_BODY_STATES')
    if not target:return
    import flash_chord.evaluation.harness as harness
    original=harness.record_object_pose_rollout
    if getattr(original,'_body_recording',False):return
    def record(env,jax_env,*args,**kwargs):
        assert env.world_count==1,'Full-body official recording must use one predetermined world'
        poses=[];assistance=[];step=jax_env.step
        def observe(action):
            output=step(action)
            poses.append(env.state_0.body_q.numpy().copy())
            assistance.append(float(env.voc_scale.numpy()[0]))
            return output
        jax_env.step=observe
        try:result=original(env,jax_env,*args,**kwargs)
        finally:jax_env.step=step
        assert len(poses)==kwargs['step_count'] and np.all(np.asarray(assistance)==0.)
        path=Path(target);path.parent.mkdir(parents=True,exist_ok=True)
        np.savez_compressed(path,body_poses=np.asarray(poses),body_names=np.asarray(env.model.body_label),
            reference_frames=np.arange(kwargs.get('start_frame',0),kwargs.get('start_frame',0)+len(poses)),
            object_assistance=np.asarray(assistance),policy_inference=True,
            checkpoint_sha256=os.environ['V2D_OFFICIAL_CHECKPOINT_SHA256'],
            trajectory_kind='Read-only full-body observations of the actual official recorded policy attempt')
        return result
    record._body_recording=True
    harness.record_object_pose_rollout=record
