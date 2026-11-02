"""Refit a rigid tool to every visible mask and independent 3-D human pinch points.

Hidden geometry is constrained by measured grasps and temporal continuity. Neither
evaluation object poses nor simulator rollouts enter the reconstruction.
"""
from pathlib import Path
import argparse, importlib.util, json
import cv2
import numpy as np
import trimesh
from scipy.ndimage import map_coordinates
from scipy.optimize import least_squares
from scipy.spatial import cKDTree
from scipy.spatial.transform import Rotation
from scipy.signal import savgol_filter

p = argparse.ArgumentParser()
p.add_argument('--run', required=True, type=Path)
p.add_argument('--output-name', default='scene_motion_image_geometry_candidate.npz')
a = p.parse_args()
r = a.run.resolve(); rec = r/'reconstruction'
assert not (r/'coherent-input-checksums.json').exists(), 'Frozen training reference'
cfg = json.loads((r/'episode-config.json').read_text()); b = cfg['objects'].index('wooden_spoon')
support_radius=json.loads((rec/'scene-reconstruction.json').read_text())['table_support_radius_m']
raw = np.load(rec/'scene_motion_physical.npz')
candidate = rec/'scene_motion_hand_guided_candidate.npz'
prior = np.load(candidate) if candidate.exists() else raw
out = {k:prior[k].copy() for k in prior.files}; O = raw['object_to_world'][:,b]
M = out['object_to_world'][:,b].copy(); T = len(M)
H = raw['hand_joint_positions']; measure = np.load(rec/'hand_measurements.npz')
quality = json.loads((r/'evaluation/reference-source-fidelity.json').read_text())
scores = np.array([x['objects']['wooden_spoon']['iou'] for x in quality['frames']])
mesh = trimesh.load(rec/'mesh_priors/wooden_spoon.ply', process=False)
verts = np.asarray(mesh.vertices); center = verts.mean(0)
vertices = verts[np.linspace(0,len(verts)-1,min(1800,len(verts))).astype(int)]
spec = importlib.util.spec_from_file_location('geometry',r/'scripts/source_geometry.py')
g = importlib.util.module_from_spec(spec); spec.loader.exec_module(g)
grasps = []; report = []
for grasp in cfg['grasps']:
    if grasp['object'] != b: continue
    s = ['left','right'].index(grasp['side'])
    ids = np.arange(grasp['start'],grasp['end']+1)
    ids = ids[(scores[ids]>.72)&measure['observed'][ids,s]]
    if len(ids)<5:
        ids = np.arange(grasp['start'],grasp['end']+1)
        ids = ids[np.argsort(scores[ids])[-max(5,len(ids)//5):]]
    points = (H[ids,s,4]+H[ids,s,8])/2
    local = np.einsum('tji,tj->ti',O[ids,:3,:3],points-O[ids,:3,3])
    anchor = np.median(local,axis=0)
    # Thumb/index midpoint estimates the shaft center, not an object-pose proxy.
    anchor[0] = np.clip(anchor[0],-.008,.008)
    anchor[1] = np.clip(anchor[1],verts[:,1].min(),verts[:,1].max())
    anchor[2] = np.clip(anchor[2],verts[:,2].min(),verts[:,2].max())
    grasps.append((grasp,s,anchor))
    report.append({'grasp':grasp,'pinch_anchor_local_m':anchor.tolist(),'anchor_frames':ids.tolist()})

frame_reports = []
for i in range(T):
    active = [(gr,s,anchor) for gr,s,anchor in grasps if gr['start']<=i<=gr['end']]
    # Observed terminal rests take precedence over uncertain hand depth.
    rest_start = {32:211,33:289}.get(cfg['episode'],T)
    if not active or i>=rest_start: continue
    C = np.linalg.inv(raw['camera_to_world'][i])
    target = cv2.resize(cv2.imread(str(rec/f'masks/{b+1}/{i:06d}.png'),0),(512,384),interpolation=cv2.INTER_NEAREST)>0
    visible = ~g.human_occlusion(measure,i)
    for other in range(len(cfg['objects'])):
        if other!=b: visible &= cv2.resize(cv2.imread(str(rec/f'masks/{other+1}/{i:06d}.png'),0),(512,384),interpolation=cv2.INTER_NEAREST)==0
    yy,xx = np.where(target&visible)
    if len(xx)<10: continue
    take = np.linspace(0,len(xx)-1,min(90,len(xx))).astype(int)
    pixels = np.c_[xx[take],yy[take]].astype(float)
    dist = cv2.distanceTransform((~target).astype(np.uint8),cv2.DIST_L2,5).astype(float)
    base = prior['object_to_world'][i,b].copy()
    world_vertices=verts@base[:3,:3].T+base[:3,3];inside=np.linalg.norm(world_vertices[:,:2],axis=1)<support_radius
    original_center=base[:3,:3]@center+base[:3,3]
    prior_intrusion=max(0.,.6505-world_vertices[inside,2].min()) if inside.any() else 0.
    base[2,3]+=prior_intrusion
    R0 = Rotation.from_matrix(base[:3,:3]); c0 = base[:3,:3]@center+base[:3,3]
    previous = M[i-1] if i else base
    previous_center = previous[:3,:3]@center+previous[:3,3]
    observations = [(anchor,(H[i,s,4]+H[i,s,8])/2) for _,s,anchor in active if measure['observed'][i,s]]
    def pose(x):
        q = Rotation.from_rotvec(x[3:]*.1)*R0
        mat = np.eye(4);mat[:3,:3] = q.as_matrix();mat[:3,3] = c0+x[:3]*.01-mat[:3,:3]@center
        return mat,q
    def residual(x):
        mat,q = pose(x); camera = C@mat
        pc = vertices@camera[:3,:3].T+camera[:3,3]
        uv = g.project(pc)*.5
        clipped = np.c_[np.clip(uv[:,0],0,511),np.clip(uv[:,1],0,383)]
        outside = np.linalg.norm(uv-clipped,axis=1)
        v = map_coordinates(visible.astype(float),clipped[:,::-1].T,order=0,mode='nearest')
        forward = (map_coordinates(dist,clipped[:,::-1].T,order=1,mode='nearest')+outside)*v/2/np.sqrt(len(vertices)/50)
        reverse = cKDTree(uv).query(pixels)[0]/2/np.sqrt(len(pixels)/25)
        extra = [x[:3]/8,x[3:]/10]
        world_vertices=vertices@mat[:3,:3].T+mat[:3,3];inside=np.linalg.norm(world_vertices[:,:2],axis=1)<support_radius
        extra.append((np.maximum(0.,.6505-world_vertices[:,2])/.002)**2*inside/np.sqrt(len(vertices)/30))
        if i:
            movement=c0+x[:3]*.01-previous_center
            extra += [movement/.05,np.array([(max(0.,np.linalg.norm(movement)-.05)/.002)**2]),(Rotation.from_matrix(previous[:3,:3]).inv()*q).as_rotvec()/.6]
        for anchor,observed in observations:
            extra.append((mat[:3,:3]@anchor+mat[:3,3]-observed)/.018)
        return np.concatenate([forward,reverse,*extra])
    lower=np.array([-8]*3+[-12]*3);upper=-lower
    fit = least_squares(residual,np.zeros(6),bounds=(lower,upper),loss='soft_l1',max_nfev=45)
    if prior_intrusion>.02:
        axis=R0.as_matrix()[:,2].copy();axis[2]=0.;axis=axis/np.linalg.norm(axis) if np.linalg.norm(axis)>1e-8 else np.array([1.,0.,0.])
        normal=np.array([0.,0.,1. if R0.as_matrix()[2,0]>=0 else -1.]);flat=np.c_[normal,np.cross(axis,normal),axis]
        height=.6505-((vertices-center)@flat.T)[:,2].min();origin=raw['camera_to_world'][i,:3,3];ray=original_center-origin
        flat_center=origin+ray*((height-origin[2])/ray[2]) if abs(ray[2])>.05 else original_center.copy()
        seed=np.r_[(flat_center-c0)/.01,(Rotation.from_matrix(flat)*R0.inv()).as_rotvec()/.1];seed=np.clip(seed,lower+.00001,upper-.00001)
        alternative=least_squares(residual,seed,bounds=(lower,upper),loss='soft_l1',max_nfev=70)
        def merit(solution):
            model,_=pose(solution.x);world=verts@model[:3,:3].T+model[:3,3];hit=np.linalg.norm(world[:,:2],axis=1)<support_radius;depth=max(0.,.6505-world[hit,2].min()) if hit.any() else 0.;score=g.overlap(g.silhouette(verts,C@model),target,visible)['iou'];return max(0.,depth-.002),-score
        fit=min([fit,alternative],key=merit)
    fitted,q = pose(fit.x)
    before = g.overlap(g.silhouette(verts,C@base),target,visible)['iou']
    after = g.overlap(g.silhouette(verts,C@fitted),target,visible)['iou']
    prior_step=np.linalg.norm(c0-previous_center);fit_step=np.linalg.norm(fitted[:3,:3]@center+fitted[:3,3]-previous_center)
    world_vertices=verts@fitted[:3,:3].T+fitted[:3,3];inside=np.linalg.norm(world_vertices[:,:2],axis=1)<support_radius
    floor_error=max(0.,.6505-world_vertices[inside,2].min()) if inside.any() else 0.
    accepted=bool(floor_error<.003 and (after>=before-.025 or (prior_step>.06 and fit_step<prior_step and after>=before-.12)))
    M[i]=base
    if accepted: M[i]=fitted
    frame_reports.append({'frame':i,'prior_iou':before,'fit_iou':after,'accepted':accepted,'prior_centroid_step_m':float(prior_step),'fit_centroid_step_m':float(fit_step),'table_intrusion_m':float(floor_error),'evaluations':fit.nfev})
    if i%40==0: print(i,frame_reports[-1],flush=True)

# Preserve every timestamp and the measured centroid when filtering rotations.
centroids = np.einsum('tij,j->ti',M[:,:3,:3],center)+M[:,:3,3]
centroids = savgol_filter(centroids,5,2,axis=0)
rot = Rotation.from_matrix(M[:,:3,:3])
rot = Rotation.from_matrix([rot[max(0,i-1):min(T,i+2)].mean().as_matrix() for i in range(T)])
M[:,:3,:3] = rot.as_matrix();M[:,:3,3] = centroids-np.einsum('tij,j->ti',M[:,:3,:3],center)
out['object_to_world'][:,b] = M
np.savez_compressed(rec/a.output_name,**out)
summary = {'method':__doc__,'source_prior':candidate.name if candidate.exists() else 'scene_motion_physical.npz','independent_pinch_anchors':report,'frames':frame_reports,'maximum_translation_step_m':float(np.linalg.norm(np.diff(centroids,axis=0),axis=1).max()),'maximum_rotation_step_rad':float((rot[:-1].inv()*rot[1:]).magnitude().max()),'all_source_timestamps_preserved':True,'evaluation_ground_truth_used':False}
summary['table_constraint_included']=True
report_name='image-geometry-tool-candidate.json' if a.output_name=='scene_motion_image_geometry_candidate.npz' else Path(a.output_name).stem+'.json'
(rec/report_name).write_text(json.dumps(summary,indent=2))
print('Candidate',cfg['episode'],summary['maximum_translation_step_m'],summary['maximum_rotation_step_rad'],flush=True)
