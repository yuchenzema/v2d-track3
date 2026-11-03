"""Source-only pose repair, multiview depth fusion and concave collision assets."""
import json,shutil,xml.etree.ElementTree as ET
from pathlib import Path
import cv2,numpy as np,trimesh,coacd
from scipy.ndimage import median_filter,distance_transform_edt,binary_fill_holes,gaussian_filter
from scipy.spatial import cKDTree
from skimage.measure import marching_cubes
from improvement_common import BASE,digest,save,utc
from source_mask_rasterizer import MeshSilhouette
from own_geometry import validate,create_urdf


def observation(rec,slot,frame):
    mask=cv2.imread(str(rec/f'masks/{slot}/{frame:06d}.png'),0)
    encoded=cv2.imread(str(rec/f'depth_foundation_stereo/{frame:06d}.png'),-1)
    mask=cv2.resize(mask,(512,384),interpolation=cv2.INTER_NEAREST)>0
    encoded=cv2.resize(encoded,(512,384),interpolation=cv2.INTER_NEAREST)
    depth=np.where(encoded>0,65535/np.maximum(encoded.astype(float),1)-1,0)
    valid=np.isfinite(depth)&(depth>.05)&(depth<3.)
    return mask,depth,valid


def image_cost(renderer,pose,mask,depth,valid):
    silhouette=renderer.render(pose);union=(mask|silhouette).sum();iou=float((mask&silhouette).sum()/max(1,union))
    predicted=renderer.depth(pose);overlap=mask&silhouette&valid&(predicted>.05)
    error=float(np.median(np.abs(predicted[overlap]-depth[overlap]))) if overlap.sum()>=30 else .05
    return 1-iou+min(.05,error)*4,dict(iou=iou,median_depth_error_m=error,observed_overlap=int(overlap.sum()))


def regularize(root,data,cfg):
    rec=root/'reconstruction';intr=json.loads((rec/'camera_intrinsics.json').read_text());objects=data['object_to_world'];camera=data['camera_to_world'];report={}
    for slot,name in enumerate(cfg['objects'],1):
        mesh=trimesh.load(rec/f'mesh_priors/{name}.ply',force='mesh',process=False);renderer=MeshSilhouette(mesh.vertices,mesh.faces,intr)
        raw=objects[:,slot-1].copy();local=np.linalg.inv(camera)@raw
        smooth=median_filter(raw[:,:3,3],size=(5,1),mode='nearest');smooth_z=median_filter(local[:,2,3],size=5,mode='nearest');rows=[]
        for frame in range(cfg['frames']):
            mask,depth,valid=observation(rec,slot,frame);baseline,old=image_cost(renderer,local[frame],mask,depth,valid)
            candidates=[local[frame]]
            moved=raw[frame].copy();moved[:3,3]=smooth[frame];candidates.append(np.linalg.inv(camera[frame])@moved)
            along_ray=local[frame].copy();z=along_ray[2,3]
            if z>.05:along_ray[:3,3]*=smooth_z[frame]/z;candidates.append(along_ray)
            best=local[frame];cost=baseline;diagnostic=old
            for pose in candidates[1:]:
                candidate,metrics=image_cost(renderer,pose,mask,depth,valid)
                if candidate<cost-.005:best,cost,diagnostic=pose,candidate,metrics
            objects[frame,slot-1]=camera[frame]@best
            rows.append(dict(frame=frame,before=old,after=diagnostic,translation_change_m=float(np.linalg.norm(objects[frame,slot-1,:3,3]-raw[frame,:3,3]))))
        report[name]=rows
    save(root/'research/source-pose-repair.json',dict(updated_utc=utc(),source_clock_unchanged=True,method='RGB-mask/depth-supported choice among raw and bounded temporal depth proposals',objects=report,ground_truth_used=False))
    return data


def fuse(mesh,views,intr):
    """Narrow-band metric TSDF; the original own surface is an explicit weak prior."""
    lo,hi=mesh.bounds;span=float(max(hi-lo));padding=max(.008,min(.03,span*.08));lo=lo-padding;hi=hi+padding
    pitch=float(max(hi-lo)/55);shape=np.maximum(4,np.ceil((hi-lo)/pitch).astype(int)+1)
    grid=np.stack(np.meshgrid(*[lo[i]+np.arange(shape[i])*pitch for i in range(3)],indexing='ij'),-1).reshape(-1,3)
    prior,_=trimesh.sample.sample_surface(mesh,12000,seed=3521)
    shell=(cKDTree(prior).query(grid)[0]<pitch*1.2).reshape(tuple(shape));inside=binary_fill_holes(shell)
    signed=(distance_transform_edt(~inside)-distance_transform_edt(inside))*pitch
    trunc=max(.008,3*pitch);volume=np.clip(signed/trunc,-1,1).reshape(-1)*.2;weights=np.full(len(grid),.2);rgb_points=[];rgb_colors=[]
    for pose,mask,depth,valid,image in views:
        xyz=grid@pose[:3,:3].T+pose[:3,3];uv=np.rint(xyz[:,:2]/np.maximum(xyz[:,2:],.001)*[intr['fx']/2,intr['fy']/2]+[intr['cx']/2,intr['cy']/2]).astype(int)
        bounded=(xyz[:,2]>.05)&(uv[:,0]>=0)&(uv[:,0]<512)&(uv[:,1]>=0)&(uv[:,1]<384);idx=np.flatnonzero(bounded);u,v=uv[idx].T
        measured=valid[v,u]&mask[v,u];idx=idx[measured];u,v=uv[idx].T;difference=depth[v,u]-xyz[idx,2]
        near=np.abs(difference)<trunc;ids=idx[near];volume[ids]+=np.clip(difference[near]/trunc,-1,1);weights[ids]+=1.
        # Observed free space in front of the surface can remove prior bumps.
        free=difference>=trunc;ids=idx[free];volume[ids]+=1.;weights[ids]+=1.
        v,u=np.where(mask&valid)
        if len(u):
            keep=np.linspace(0,len(u)-1,min(2000,len(u)),dtype=int);u,v=u[keep],v[keep];z=depth[v,u]
            points=np.c_[(u-intr['cx']/2)/(intr['fx']/2)*z,(v-intr['cy']/2)/(intr['fy']/2)*z,z]
            rgb_points.append((points-pose[:3,3])@pose[:3,:3]);rgb_colors.append(image[v,u])
    field=gaussian_filter((volume/weights).reshape(tuple(shape)),.55)
    if field.min()>=0 or field.max()<=0:raise ValueError('Multiview depth did not define a signed surface')
    vertices,faces,_,_=marching_cubes(field,level=0,spacing=(pitch,)*3);vertices+=lo
    candidate=trimesh.Trimesh(vertices,faces,process=True);candidate=max(candidate.split(only_watertight=False),key=lambda part:part.area)
    if len(candidate.faces)>3600:candidate=candidate.simplify_quadric_decimation(face_count=3600)
    if len(candidate.faces)<32 or not np.isfinite(candidate.vertices).all() or candidate.area<=0:raise ValueError('Invalid fused surface')
    color_points=np.concatenate(rgb_points) if rgb_points else mesh.vertices;color_values=np.concatenate(rgb_colors) if rgb_colors else mesh.visual.vertex_colors[:,:3]
    candidate.visual.vertex_colors=color_values[cKDTree(color_points).query(candidate.vertices)[1]]
    candidate.fix_normals();return candidate


def collision_assets(root,name,mesh):
    """Convex pieces from the own surface, retaining handle/rack openings."""
    assets=root/'reconstruction/mesh_priors';coacd.set_log_level('error')
    parts=coacd.run_coacd(coacd.Mesh(np.asarray(mesh.vertices,dtype=np.float64),np.asarray(mesh.faces,dtype=np.int32)),
        threshold=.04,max_convex_hull=16,preprocess_resolution=30,resolution=1200,mcts_iterations=30,seed=443)
    assert parts,'No own collision decomposition'
    create_urdf(assets/f'{name}.urdf',name,mesh);tree=ET.parse(assets/f'{name}.urdf');link=tree.getroot().find('link')
    for node in list(link.findall('collision')):link.remove(node)
    suffixes=[]
    for i,(vertices,faces) in enumerate(parts):
        suffix=f'collision_{i:03d}.obj';path=assets/f'{name}.{suffix}'
        trimesh.Trimesh(vertices,faces,process=False).export(path,include_color=False)
        geometry=ET.SubElement(ET.SubElement(link,'collision'),'geometry');ET.SubElement(geometry,'mesh',filename='/episode/reconstruction/mesh_priors/'+path.name);suffixes.append(suffix)
    tree.write(assets/f'{name}.urdf');return suffixes


def prepare(parent,episode):
    parent=Path(parent);validate(parent);original=json.loads((parent/'episode-config.json').read_text())
    root=BASE/f'runs/evaluation_episode_{episode:06d}_quality_20261103_v1'
    if (root/'research/quality-source-completed.json').exists():validate(root);return root
    proof_path=root/'research/own-geometry-provenance.json';progress_path=root/'research/quality-source-progress.json'
    if proof_path.exists() and progress_path.exists():
        existing=json.loads(proof_path.read_text());progress=json.loads(progress_path.read_text())
        if existing.get('revision')=='evaluation_rgb_depth_multiview_v4_20261103' and set(progress.get('objects',{}))==set(original['objects']):
            validate(root);save(root/'research/quality-source-completed.json',dict(progress,recovered_completed_source=True));return root
    for folder in ['reconstruction','reconstruction/mesh_priors','research']:(root/folder).mkdir(parents=True,exist_ok=True)
    rec=root/'reconstruction';source=parent/'reconstruction'
    for name in ['frames','masks','depth_foundation_stereo']:
        if not (rec/name).exists():(rec/name).symlink_to((source/name).resolve(),target_is_directory=True)
    for name in ['camera_intrinsics.json','hand_measurements.npz','table-camera-calibration.json','table-camera-calibration.npz','scene-reconstruction.json']:
        if (source/name).exists() and not (rec/name).exists():shutil.copy2(source/name,rec/name)
    for path in (source/'mesh_priors').iterdir():
        if path.is_file() and not (rec/'mesh_priors'/path.name).exists():shutil.copy2(path,rec/'mesh_priors'/path.name)
    cfg=dict(original,run=str(root));save(root/'episode-config.json',cfg)
    data_file=source/'scene_motion_physical.npz'
    with np.load(data_file) as raw:data={k:raw[k].copy() for k in raw.files}
    data=regularize(root,data,cfg);np.savez_compressed(rec/'scene_motion_physical.npz',**data)
    proof=json.loads((parent/'research/own-geometry-provenance.json').read_text());intr=json.loads((rec/'camera_intrinsics.json').read_text());summary={}
    for slot,name in enumerate(cfg['objects'],1):
        old=trimesh.load(rec/f'mesh_priors/{name}.ply',force='mesh',process=False);renderer=MeshSilhouette(old.vertices,old.faces,intr);ranked=[]
        for frame in np.linspace(0,cfg['frames']-1,min(24,cfg['frames']),dtype=int):
            pose=np.linalg.inv(data['camera_to_world'][frame])@data['object_to_world'][frame,slot-1]
            mask,depth,valid=observation(rec,slot,int(frame));cost,metrics=image_cost(renderer,pose,mask,depth,valid)
            if (mask&valid).sum()<80:continue
            image=cv2.cvtColor(cv2.resize(cv2.imread(str(rec/f'frames/{frame:06d}.png')),(512,384)),cv2.COLOR_BGR2RGB)
            ranked.append((cost,int(frame),(pose,mask,depth,valid,image)))
        ranked.sort(key=lambda row:row[0]);fit=ranked[::2][:8];held=ranked[1::2][:8];candidate=old;accepted=False;error=None;before=None;after=None
        if len(fit)>=2 and held:
            try:
                proposed=fuse(old,[r[2] for r in fit],intr);new_renderer=MeshSilhouette(proposed.vertices,proposed.faces,intr)
                before=float(np.mean([r[0] for r in held]));after=float(np.mean([image_cost(new_renderer,*r[2][:4])[0] for r in held]))
                if after<before-.005:candidate=proposed;accepted=True
                else:
                    path=root/f'research/fusion-rejected-{name}.glb';proposed.export(path)
            except Exception as failure:error=str(failure)
        assets=rec/'mesh_priors'
        for suffix in ['ply','glb','obj']:candidate.export(assets/f'{name}.{suffix}')
        suffixes=collision_assets(root,name,candidate)
        row=proof['objects'][name];row['mesh_sha256']={suffix:digest(assets/f'{name}.{suffix}') for suffix in ['ply','glb','obj','urdf',*suffixes]}
        row.update(watertight=bool(candidate.is_watertight),faces=len(candidate.faces),vertices=len(candidate.vertices))
        row['inputs'][str(data_file)]=digest(data_file)
        for _,frame,_ in ranked:
            for relative in [f'masks/{slot}/{frame:06d}.png',f'depth_foundation_stereo/{frame:06d}.png',f'frames/{frame:06d}.png']:
                path=(source/relative).resolve();row['inputs'][str(path)]=digest(path)
        summary[name]=dict(multiview_geometry_selected=accepted,fit_frames=[r[1] for r in fit],validation_frames=[r[1] for r in held],before_source_cost=before,after_source_cost=after,collision_hulls=len(suffixes),fusion_error=error)
        save(root/'research/quality-source-progress.json',dict(updated_utc=utc(),objects=summary))
    proof.update(revision='evaluation_rgb_depth_multiview_v4_20261103',builder_sha256=digest(Path(__file__)),updated_utc=utc(),
        old_object_poses_used=True,own_parent_run=str(parent),own_parent_provenance_sha256=digest(parent/'research/own-geometry-provenance.json'),
        historical_scan_based_poses_used=False,geometry_method=__doc__,quality_passed=False)
    save(root/'research/own-geometry-provenance.json',proof);validate(root)
    save(root/'research/quality-source-completed.json',dict(updated_utc=utc(),objects=summary,own_previous_estimates_used=True,public_scans_used=False))
    return root
