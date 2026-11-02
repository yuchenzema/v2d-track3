"""One robust static-table anchor for an independently measured metric camera path."""
from pathlib import Path
import json
import numpy as np
from improvement_common import save,utc


def anchored_cameras(slam,centers,normals,scores,height=.65):
    world_centers=np.einsum('tij,tj->ti',slam[:,:3,:3],centers)+slam[:,:3,3]
    world_normals=np.einsum('tij,tj->ti',slam[:,:3,:3],normals)
    valid=np.isfinite(world_centers).all(axis=1)&np.isfinite(world_normals).all(axis=1)&(np.asarray(scores)>=.15)
    if valid.sum()<8:raise RuntimeError('Too few independent table measurements for a static world anchor')
    center=np.median(world_centers[valid],axis=0)
    distances=np.linalg.norm(world_centers-center,axis=1)
    threshold=max(.02,3.*float(np.median(distances[valid])))
    inliers=valid&(distances<=threshold)
    if inliers.sum()<8:raise RuntimeError('Static table center lacks multi-frame consensus')
    center=np.median(world_centers[inliers],axis=0)
    normal=np.median(world_normals[inliers],axis=0);normal/=np.linalg.norm(normal)
    x=slam[0,:3,0].copy();x-=normal*(x@normal);x/=np.linalg.norm(x)
    global_frame=np.eye(4);global_frame[:3,:3]=np.stack([x,np.cross(normal,x),normal]);global_frame[:3,3]=np.array([0.,0.,height])-global_frame[:3,:3]@center
    cameras=global_frame[None]@slam
    return cameras,dict(frames=int(len(slam)),table_anchor_inlier_frames=np.flatnonzero(inliers).tolist(),
        rejected_table_measurement_frames=np.flatnonzero(~inliers).tolist(),consensus_threshold_m=threshold,
        measured_center_residual_median_m=float(np.median(np.linalg.norm(world_centers[inliers]-center,axis=1))),
        camera_path_preserved_by_single_rigid_transform=True,per_frame_table_jitter_not_added_to_camera=True,
        anchor_center_in_odometry_m=center.tolist(),anchor_normal_in_odometry=normal.tolist())


def run(root):
    root=Path(root);rec=root/'reconstruction';cfg=json.loads((root/'episode-config.json').read_text())
    if cfg.get('camera_odometry_method')!='sift_pnp_metric':return
    report=rec/'metric-table-anchor.json'
    if report.exists():return
    measurements=np.load(rec/'table-camera-calibration.npz');values={k:measurements[k] for k in measurements.files}
    slam=np.load(rec/'sift-pnp-odometry.npz')['camera_to_world']
    cameras,evidence=anchored_cameras(slam,values['table_centers_camera'],values['table_planes_camera'][:,:3],values['circle_inlier_fraction'])
    original=rec/'table-camera-local-circle.npz'
    if not original.exists():np.savez_compressed(original,**values)
    values['camera_to_world']=cameras
    np.savez_compressed(rec/'table-camera-calibration.npz',**values)
    metadata=json.loads((rec/'table-camera-calibration.json').read_text());metadata.update(
        yaw_source='Background SIFT + metric stereo PnP-RANSAC; one robust global table-plane rotation',
        translation_source='Metric SIFT/PnP camera path, transformed once into a multi-frame static-table coordinate frame',
        depth_source='FoundationStereo metric C/B depth registered to RGB; missing depths excluded',
        static_table_bundle=evidence)
    save(rec/'table-camera-calibration.json',metadata)
    save(report,dict(updated_utc=utc(),method='Multi-frame static-table consensus; one global rigid transform of metric background odometry',ground_truth_used=False,**evidence))
