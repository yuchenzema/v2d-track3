"""Approximate source poses from measured mask/depth clouds and own mesh ICP.

This submission-priority estimator replaces slow FoundationPose registration for
unfinished episodes. Its fits and unobserved-frame priors remain diagnostics.
It never generates a policy rollout or reads released object scans/poses.
"""
import json
from pathlib import Path
import cv2
import numpy as np
import trimesh
from scipy.spatial import cKDTree
from scipy.spatial.transform import Rotation
from improvement_common import digest,save,utc


def rigid_fit(source,target,initial,iterations=5):
    """Trimmed rigid ICP, with metre scale fixed by source stereo measurements."""
    pose=np.array(initial,copy=True);tree=cKDTree(target)
    pose[:3,3]+=np.median(target,axis=0)-np.median(source@pose[:3,:3].T+pose[:3,3],axis=0)
    for _ in range(iterations):
        moved=source@pose[:3,:3].T+pose[:3,3];distance,index=tree.query(moved)
        valid=distance<=max(.003,float(np.quantile(distance,.8)))
        if valid.sum()<8:break
        x,y=moved[valid],target[index[valid]];cx,cy=x.mean(0),y.mean(0)
        u,_,vt=np.linalg.svd((x-cx).T@(y-cy));rotation=vt.T@u.T
        if np.linalg.det(rotation)<0:vt[-1]*=-1;rotation=vt.T@u.T
        delta=np.eye(4);delta[:3,:3]=rotation;delta[:3,3]=cy-rotation@cx;pose=delta@pose
    assert np.isfinite(pose).all() and np.allclose(pose[:3,:3].T@pose[:3,:3],np.eye(3),atol=1e-6)
    residual=float(np.median(tree.query(source@pose[:3,:3].T+pose[:3,3])[0]))
    return pose,residual


def measured_cloud(rec,slot,frame,intr):
    mask=cv2.imread(str(rec/f'masks/{slot}/{frame:06d}.png'),0)
    encoded=cv2.imread(str(rec/f'depth_foundation_stereo/{frame:06d}.png'),-1)
    if mask is None or encoded is None:raise ValueError(f'Missing source mask/depth at frame {frame}')
    height,width=mask.shape;size=(min(width,320),round(height*min(width,320)/width))
    mask=cv2.resize(mask,size,interpolation=cv2.INTER_NEAREST)>0
    encoded=cv2.resize(encoded,size,interpolation=cv2.INTER_NEAREST)
    z=65535/np.maximum(encoded.astype(float),1)-1;valid=mask&(encoded>0)&np.isfinite(z)&(z>.05)&(z<3.)
    if valid.sum()<30:return np.empty((0,3))
    median=np.median(z[valid]);mad=np.median(np.abs(z[valid]-median));valid&=np.abs(z-median)<max(.012,6*mad)
    v,u=np.where(valid);depth=z[valid];x=u*width/size[0];y=v*height/size[1]
    points=np.c_[(x-intr['cx'])/intr['fx']*depth,(y-intr['cy'])/intr['fy']*depth,depth]
    return points[np.linspace(0,len(points)-1,min(640,len(points)),dtype=int)] if len(points) else points


def run(root,cfg,name,slot):
    root=Path(root);rec=root/'reconstruction';folder=rec/'object_poses'/name;folder.mkdir(exist_ok=True)
    proof=json.loads((root/'research/own-geometry-provenance.json').read_text());geometry=proof['objects'][name]
    intr=json.loads((rec/'camera_intrinsics.json').read_text());mesh=trimesh.load(rec/f'mesh_priors/{name}.ply',force='mesh',process=False)
    points=mesh.vertices[np.linspace(0,len(mesh.vertices)-1,min(384,len(mesh.vertices)),dtype=int)]
    camera_path=rec/'sift-pnp-odometry.npz';cameras=np.load(camera_path)['camera_to_world']
    identity=dict(method='source_metric_trimmed_ICP_v1',own_mesh_sha256=digest(rec/f'mesh_priors/{name}.ply'),
        camera_sha256=digest(camera_path),intrinsics_sha256=digest(rec/'camera_intrinsics.json'),code_sha256=digest(Path(__file__)),
        iterations=5,source_points=len(points),public_object_geometry_used=False)
    marker=folder.with_suffix('.metric-icp-identity.json')
    if marker.exists():assert json.loads(marker.read_text())==identity,'Source ICP identity changed; preserve prior estimates'
    else:
        assert not list(folder.glob('*.json')),'Refuse to mix another pose estimator into this sequence'
        save(marker,identity)
    reference=int(geometry['reference_frame']);initial=np.asarray(geometry['initial_object_to_camera']);rows=[]
    for direction in [list(range(reference,cfg['frames'])),list(range(reference-1,-1,-1))]:
        previous=initial.copy();previous_frame=reference
        for frame in direction:
            target=folder/f'{frame:06d}.json'
            mask=rec/f'masks/{slot}/{frame:06d}.png';depth=rec/f'depth_foundation_stereo/{frame:06d}.png'
            inputs=dict(mask_sha256=digest(mask),depth_sha256=digest(depth))
            if target.exists():
                saved=json.loads(target.read_text());assert saved['source_inputs']==inputs
                pose=np.eye(4);pose[:3,:3]=Rotation.from_quat(saved['rotation'],scalar_first=True).as_matrix();pose[:3,3]=saved['translation']
                row=saved['fit_diagnostic']
            else:
                seed=np.linalg.inv(cameras[frame])@cameras[previous_frame]@previous
                measured=measured_cloud(rec,slot,frame,intr)
                if frame==reference:pose=initial.copy();residual=None
                elif len(measured)>=30:pose,residual=rigid_fit(points,measured,seed)
                else:pose=seed;residual=None
                row=dict(frame=frame,measured_points=len(measured),median_fit_distance_m=residual,
                    unobserved_frame_motion_prior=len(measured)<30,strict_alignment_verified=False)
                save(target,dict(rotation=Rotation.from_matrix(pose[:3,:3]).as_quat(scalar_first=True).tolist(),
                    translation=pose[:3,3].tolist(),scale=[1.,1.,1.],source_inputs=inputs,fit_diagnostic=row))
            rows.append(row);previous=pose;previous_frame=frame
            if len(rows)%25==0:print('Approximate metric source ICP',name,len(rows),'/',cfg['frames'],flush=True)
    save(root/f'research/metric-icp-{name}.json',dict(identity,updated_utc=utc(),frames=cfg['frames'],
        quality_is_diagnostic=True,policy_rollout=False,per_frame=sorted(rows,key=lambda row:row['frame'])))
