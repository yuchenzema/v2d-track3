"""Reconstruct metric object surfaces from evaluation pixels, never supplied scans.

Visible surfaces use measured depth and RGB. Unseen backs have an explicit
constant-thickness completion prior; this is an estimate, not recovered detail.
The same completed mesh must be used for tracking, simulation and submission.
"""
import json
from pathlib import Path
import xml.etree.ElementTree as ET

import cv2
import numpy as np
import trimesh
from scipy.ndimage import distance_transform_edt
from improvement_common import BASE, BATCH, digest, save, utc

REVISION='evaluation_rgb_depth_geometry_v3_20261102'


def required(out=None,base=None):
    folder=Path(out) if out is not None else (Path(base)/'runs/evaluation_batch_20261015/focused_episodes_20261028' if base is not None else BATCH/'focused_episodes_20261028')
    path=folder/'own-geometry-policy.json'
    return path.is_file() and json.loads(path.read_text()).get('enabled') is True


def reject_forbidden_path(path):
    parts=Path(path).resolve().parts
    if any(parts[i]=='public' and parts[i+1] in {'mesh','urdf','data'} for i in range(len(parts)-1)):
        raise ValueError('Public object geometry/trajectories are forbidden reconstruction inputs: '+str(path))


def reconstruct_surface(rgb, mask, depth, intr, face_budget=3600,require_watertight=True):
    """Build two manifold pixel-grid surfaces joined only along their boundary."""
    height,width=mask.shape
    size=(min(width,320),max(1,round(height*min(width,320)/width)))
    m=cv2.resize(mask.astype(np.uint8),size,interpolation=cv2.INTER_NEAREST)>0
    z=cv2.resize(depth.astype(np.float32),size,interpolation=cv2.INTER_NEAREST)
    colors=cv2.resize(rgb,size,interpolation=cv2.INTER_AREA)
    measured=m & np.isfinite(z) & (z>.05) & (z<3.)
    if measured.sum()<80 or measured.sum()<.25*m.sum():
        raise ValueError('Insufficient metric source depth for this object surface')
    values=z[measured];median=float(np.median(values));mad=float(np.median(np.abs(values-median)))
    measured &= np.abs(z-median)<max(.012,6*mad)
    if measured.sum()<80:raise ValueError('Metric depth support vanished after robust outlier rejection')
    # Fill only inside the observed silhouette; missing depths remain explicit
    # estimates in the provenance and never become measured source evidence.
    nearest=distance_transform_edt(~measured,return_distances=False,return_indices=True)
    filled=z[tuple(nearest)]
    ids=np.full(m.shape,-1,np.int32);ids[m]=np.arange(m.sum())
    top_left=ids[:-1,:-1];top_right=ids[:-1,1:];bottom_left=ids[1:,:-1];bottom_right=ids[1:,1:]
    faces=np.concatenate([np.stack([top_left,bottom_left,top_right],-1).reshape(-1,3),
                          np.stack([top_right,bottom_left,bottom_right],-1).reshape(-1,3)])
    faces=faces[(faces>=0).all(1)]
    if len(faces)<50:raise ValueError('Source silhouette has insufficient connected surface')
    v,u=np.where(m);zz=filled[m]
    sx,sy=size[0]/width,size[1]/height
    points=np.c_[(u/sx-intr['cx'])/intr['fx']*zz,(v/sy-intr['cy'])/intr['fy']*zz,zz]
    # This prior is intentionally independent of object identity/public scans.
    # Depth variation gives a conservative bound; the unseen side is uncertain.
    span=float(max(np.ptp(points[:,0]),np.ptp(points[:,1])))
    # Camera-z variation mostly measures slant on a plate. Estimate the unseen
    # completion from surface-normal scatter instead, with a conservative
    # object-size bound; keep its uncertainty explicit in the proof.
    observed=points[np.isfinite(points).all(1)]
    _,_,basis=np.linalg.svd(observed-np.median(observed,axis=0),full_matrices=False)
    normal=(observed-np.median(observed,axis=0))@basis[-1]
    scatter=2*float(np.median(np.abs(normal-np.median(normal))))
    thickness=float(np.clip(scatter,.003,max(.003,min(.04,.1*span))))
    backside=points.copy();backside[:,2]+=thickness
    count=len(points)
    edges=np.concatenate([faces[:,[0,1]],faces[:,[1,2]],faces[:,[2,0]]])
    _,inverse,counts=np.unique(np.sort(edges,axis=1),axis=0,return_inverse=True,return_counts=True)
    boundary=edges[counts[inverse]==1]
    sides=np.concatenate([np.c_[boundary[:,0],boundary[:,1],boundary[:,0]+count],
                          np.c_[boundary[:,1],boundary[:,1]+count,boundary[:,0]+count]])
    allfaces=np.concatenate([faces,faces[:,::-1]+count,sides])
    center=np.median(points,axis=0);vertices=np.concatenate([points,backside])-center
    color=colors[m];color=np.concatenate([color,color])
    mesh=trimesh.Trimesh(vertices,allfaces,vertex_colors=color,process=True)
    # Small disconnected segmentation speckles do not define another object.
    components=mesh.split(only_watertight=False)
    mesh=max(components,key=lambda component:component.area)
    mesh.fix_normals()
    if not mesh.is_watertight:raise ValueError('Source surface completion is not watertight')
    raw_faces=len(mesh.faces)
    if raw_faces>face_budget:
        old=mesh
        mesh=mesh.simplify_quadric_decimation(face_count=face_budget)
        from scipy.spatial import cKDTree
        indices=cKDTree(old.vertices).query(mesh.vertices)[1]
        mesh.visual.vertex_colors=old.visual.vertex_colors[indices]
    if ((require_watertight and not mesh.is_watertight) or len(mesh.faces)>face_budget
            or not np.isfinite(mesh.vertices).all() or len(mesh.faces)<4 or not mesh.area>0):
        raise ValueError('Budgeted own reconstruction failed finite/topology checks')
    initial=np.eye(4);initial[:3,3]=center
    return mesh,initial,dict(observed_surface_depth_pixels=int(measured.sum()),
        silhouette_pixels=int(m.sum()),estimated_depth_fraction=1-float(measured.sum()/m.sum()),
        unseen_back_thickness_prior_m=thickness,raw_faces=raw_faces,
        vertices=len(mesh.vertices),faces=len(mesh.faces),watertight=bool(mesh.is_watertight),
        watertight_acceptance_required=require_watertight,
        hidden_shape_recovered=False)


def candidate_frames(root,cfg,body):
    rec=root/'reconstruction';ranked=[]
    for frame in range(cfg['frames']):
        mask=cv2.imread(str(rec/f'masks/{body+1}/{frame:06d}.png'),0)
        if mask is None:raise ValueError('A source mask is missing')
        area=int(np.count_nonzero(mask))
        if area<100:continue
        dpath=rec/f'depth_foundation_stereo/{frame:06d}.png'
        encoded=cv2.imread(str(dpath),cv2.IMREAD_UNCHANGED) if dpath.is_file() else None
        if encoded is None:continue
        measured=int(np.count_nonzero((mask>0)&(encoded>0)))
        # Prefer well-observed, large views. No ground truth, score or old pose.
        ranked.append((measured*min(1.,measured/area),frame))
    return [frame for _,frame in sorted(ranked,reverse=True)[:12]]


def create_urdf(path,name,mesh):
    robot=ET.Element('robot',name=name);link=ET.SubElement(robot,'link',name='object')
    mass=float(np.clip(abs(mesh.volume)*600.,.015,.8))
    inertia=ET.SubElement(link,'inertial');ET.SubElement(inertia,'origin',xyz='0 0 0');ET.SubElement(inertia,'mass',value=str(mass))
    diagonal=mass*(np.sum(mesh.extents**2)-mesh.extents**2)/12
    ET.SubElement(inertia,'inertia',ixx=str(diagonal[0]),iyy=str(diagonal[1]),izz=str(diagonal[2]),ixy='0',ixz='0',iyz='0')
    for kind in ['visual','collision']:
        geometry=ET.SubElement(ET.SubElement(link,kind),'geometry')
        ET.SubElement(geometry,'mesh',filename=f'/episode/reconstruction/mesh_priors/{name}.obj')
    ET.ElementTree(robot).write(path)


def build(root,cfg):
    root=Path(root);rec=root/'reconstruction';assets=rec/'mesh_priors';assets.mkdir(exist_ok=True)
    proof=root/'research/own-geometry-provenance.json'
    pending=root/'research/own-geometry-progress.json'
    if proof.is_file():return validate(root)
    objects=json.loads(pending.read_text()).get('objects',{}) if pending.is_file() else {}
    if any(assets.iterdir()) and not pending.is_file():raise ValueError('Own geometry requires a fresh mesh directory; preserve historical assets')
    intr=json.loads((rec/'camera_intrinsics.json').read_text())
    save(pending,dict(updated_utc=utc(),objects=objects))
    for body,name in enumerate(cfg['objects']):
        if name in objects:
            for suffix,expected in objects[name]['mesh_sha256'].items():
                if digest(assets/f'{name}.{suffix}')!=expected:raise ValueError('Interrupted own geometry artifact changed')
            continue
        errors=[]
        for frame in candidate_frames(root,cfg,body):
            image=rec/f'frames/{frame:06d}.{cfg.get("frame_cache_format","png")}'
            mask=rec/f'masks/{body+1}/{frame:06d}.png';depth=rec/f'depth_foundation_stereo/{frame:06d}.png'
            rgb=cv2.cvtColor(cv2.imread(str(image)),cv2.COLOR_BGR2RGB)
            encoded=cv2.imread(str(depth),cv2.IMREAD_UNCHANGED)
            z=np.where(encoded>0,65535/np.maximum(encoded.astype(float),1)-1,0)
            try:mesh,pose,details=reconstruct_surface(rgb,cv2.imread(str(mask),0)>0,z,intr,cfg.get('submission_mesh_face_budget',3600),require_watertight=not cfg.get('fast_submission',False))
            except ValueError as error:errors.append(dict(frame=frame,error=str(error)));continue
            for suffix in ['ply','glb','obj']:mesh.export(assets/f'{name}.{suffix}')
            create_urdf(assets/f'{name}.urdf',name,mesh)
            objects[name]=dict(reference_frame=frame,initial_object_to_camera=pose.tolist(),**details,
                inputs={str(p):digest(p) for p in [image,mask,depth,rec/'camera_intrinsics.json',Path(cfg['source'])]},
                mesh_sha256={suffix:digest(assets/f'{name}.{suffix}') for suffix in ['ply','glb','obj','urdf']},
                rejected_source_views=errors)
            save(pending,dict(updated_utc=utc(),objects=objects))
            print('Own source geometry',cfg['episode'],name,'frame',frame,'faces',len(mesh.faces),flush=True)
            break
        if name not in objects:raise RuntimeError(f'No source view can reconstruct {name}: {errors}')
    save(proof,dict(revision=REVISION,updated_utc=utc(),episode=cfg['episode'],objects=objects,
        builder_sha256=digest(Path(__file__)),public_object_meshes_used=False,public_object_textures_used=False,
        public_marker_templates_used=False,old_object_poses_used=False,
        geometry_method=__doc__,quality_passed=False,policy_training_started=False))
    return validate(root)


def validate(root,training_root=None):
    root=Path(root);proof=root/'research/own-geometry-provenance.json'
    if not proof.is_file():raise ValueError('Missing own-reconstruction provenance; public-scan-derived runs are ineligible')
    data=json.loads(proof.read_text())
    if data.get('revision') not in [REVISION,'evaluation_rgb_depth_multiview_v4_20261103']:raise ValueError('Unknown own-geometry reconstruction version')
    if any(data.get(k) is not False for k in ['public_object_meshes_used','public_object_textures_used','public_marker_templates_used']):
        raise ValueError('Forbidden object prior in reconstruction lineage')
    if data.get('old_object_poses_used') is not False:
        if not (data.get('revision')=='evaluation_rgb_depth_multiview_v4_20261103' and data.get('historical_scan_based_poses_used') is False):raise ValueError('Forbidden object pose prior')
        parent=Path(data['own_parent_run'])
        if parent.resolve()==root.resolve() or digest(parent/'research/own-geometry-provenance.json')!=data['own_parent_provenance_sha256']:raise ValueError('Invalid own reconstruction parent')
        validate(parent)
    cfg=json.loads((root/'episode-config.json').read_text())
    if cfg.get('dataset_split')!='evaluation' or not cfg.get('own_geometry_required'):
        raise ValueError('Own reconstruction must be bound to the evaluation source')
    if set(data.get('objects',{}))!=set(cfg['objects']):raise ValueError('Incomplete own-geometry object roster')
    for name,row in data['objects'].items():
        if not row.get('inputs') or not row.get('mesh_sha256'):raise ValueError('Missing measured inputs or mesh hashes')
        for path,expected in row['inputs'].items():
            reject_forbidden_path(path)
            if digest(path)!=expected:raise ValueError('Own-geometry source changed: '+path)
        for suffix,expected in row['mesh_sha256'].items():
            path=Path(training_root or root)/f'reconstruction/mesh_priors/{name}.{suffix}'
            if digest(path)!=expected:raise ValueError('Own geometry changed after reconstruction: '+str(path))
    return data


if __name__=='__main__':
    import argparse
    parser=argparse.ArgumentParser();parser.add_argument('--run',type=Path,required=True);args=parser.parse_args()
    build(args.run,json.loads((args.run/'episode-config.json').read_text()))
