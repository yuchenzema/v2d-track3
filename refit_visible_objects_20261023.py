"""Bounded rigid pose refinement against visible original-video object pixels."""
import argparse
import json
import re
import hashlib
import subprocess
from pathlib import Path
import cv2
import numpy as np
import trimesh
from scipy.ndimage import distance_transform_edt,map_coordinates
from scipy.optimize import minimize
from scipy.spatial.transform import Rotation
from improvement_common import BASE,BATCH,save,utc
from sphere_marker_reference import marker_residual
from source_pose_seed import rebase_reconstructed_pose
from source_table_constraint import constrain_pose

p=argparse.ArgumentParser();p.add_argument('--run',type=Path,required=True);p.add_argument('--request',type=Path,required=True);a=p.parse_args()
r=a.run;assert not (r/'coherent-input-checksums.json').exists()
c=json.loads((r/'episode-config.json').read_text())
policy_path=BATCH/'focused_episodes_20261028/marker-reference-policy.json'
sphere_policy=json.loads(policy_path.read_text()) if policy_path.exists() else {}
use_spheres=sphere_policy.get('enabled') and c['episode'] in sphere_policy.get('episodes',[])
marker=r/'research/source-pixel-pose-refinement.json'
if use_spheres:marker=r/'research/source-pixel-pose-refinement-v2-spheres.json'
if marker.exists():raise SystemExit(0)
job=json.loads(a.request.read_text());rec=r/'reconstruction'
sphere_frames=None;sphere_templates={};sphere_summary={}
if use_spheres:
    subprocess.run([str(r/'.venv-video/bin/python'),str(BASE/'track_object_spheres.py'),'--run',str(r)],check=True)
    sphere_frames=json.loads((rec/'locating-sphere-observations.json').read_text())['frames']
    sphere_summary=json.loads((r/'research/locating-sphere-tracking.json').read_text())['objects']
    sphere_templates=json.loads((BATCH/'focused_episodes_20261028/huggingface_audit_20261028/marker-templates.json').read_text())['objects']
loaded=np.load(rec/'scene_motion_physical.npz');motion={k:loaded[k].copy() for k in loaded.files}
np.savez_compressed(rec/'scene_motion_before_pixel_refinement.npz',**motion)
hands=np.load(rec/'hand_measurements.npz');intr=json.loads((rec/'camera_intrinsics.json').read_text())
windows=job.get('worst_windows',[])[:3]
all_windows=job.get('worst_windows',[])
if all_windows:
    start=(max(1,job.get('generation',1))-1)%len(all_windows)
    windows=(all_windows+all_windows)[start:start+3]
indexes={i for w in windows for i in range(w['start'],min(c['frames'],w['end']+1))}
seed_motion=None
if job.get('seed_reconstruction_run'):
    seed_file=Path(job['seed_reconstruction_run'])/'reconstruction/scene_motion_physical.npz'
    if seed_file.is_file():
        seed_motion=np.load(seed_file)
        assert seed_motion['object_to_world'].shape==motion['object_to_world'].shape
# These initial table supports were observed in the original human views.
# In episode30 the brush rests in the rack, so it is deliberately excluded.
table_supported={19:['mini_sweeper','mini_dust_pan'],20:['water_pitcher'],30:['plastic_dish_rack'],33:['blue_cup']}.get(c['episode'],[])
support_ends={name:min((g['start'] for g in c['grasps'] if c['objects'][g['object']]==name),default=0)-4 for name in table_supported}
for end in support_ends.values():indexes.update(range(max(0,end+1)))
refit_indexes=set(indexes);indexes=list(range(c['frames'])) if seed_motion is not None else sorted(indexes)
records=[];previous={};meshes={name:trimesh.load(rec/f'mesh_priors/{name}.ply',process=False) for name in c['objects']}
table=None;collision_vertices={}
support_files=list((r/'human_motion_data/track3_collision_variant/reconstructed_stage').glob('*support.usda'))
if support_files:
    support=support_files[0].read_text()
    radius=float(re.search(r'double radius = ([0-9.e+-]+)',support)[1])
    height=float(re.search(r'double height = ([0-9.e+-]+)',support)[1])
    center=np.array([float(x) for x in re.search(r'xformOp:translate = \(([^)]+)\)',support)[1].split(',')])
    table=(center,radius,float(center[2]+height/2+.0005))
elif (rec/'table-camera-calibration.json').exists():
    calibration=json.loads((rec/'table-camera-calibration.json').read_text())
    table=(np.r_[calibration['support_center_xy_m'],calibration['support_height_gauge_m']],
           float(calibration['support_radius_m']),float(calibration['support_height_gauge_m'])+.0005)
hull_file=r/'evaluation/object-collision-hulls.npz'
if hull_file.exists():
    with np.load(hull_file) as hulls:
        for name in c['objects']:
            parts=[hulls[k] for k in hulls.files if k.startswith(name+'__')]
            if parts:collision_vertices[name]=np.concatenate(parts)
for frame in indexes:
    camera=np.linalg.inv(motion['camera_to_world'][frame]);exclude=np.zeros((384,512),np.uint8)
    for side in range(2):
        if not hands['observed'][frame,side]:continue
        points=np.rint(hands['uv'][frame,side]*.5).astype(np.int32)
        cv2.fillConvexPoly(exclude,cv2.convexHull(points),1)
    masks=[cv2.resize(cv2.imread(str(rec/f'masks/{b+1}/{frame:06d}.png'),0),(512,384),interpolation=cv2.INTER_NEAREST)>0 for b in range(len(c['objects']))]
    for b,name in enumerate(c['objects']):
        sphere_points=[];sphere_pixels=[]
        if sphere_frames is not None and name in sphere_templates:
            stats=sphere_summary[name]['observations_per_marker']
            for observation in sphere_frames[frame]['objects'][name]['markers']:
                if observation['observed'] and stats.get(str(observation['marker_id']),0)>=8 and observation['measurement']['cost']<4.5:
                    sphere_points.append(sphere_templates[name]['markers'][observation['marker_id']]['center_m'])
                    sphere_pixels.append(observation['measurement']['xy'])
        def sphere_error(M):
            if not sphere_points:return None
            return float(np.median(np.linalg.norm(marker_residual(camera@M,sphere_points,sphere_pixels,intr),axis=1)))
        valid=exclude==0
        for other,mask in enumerate(masks):
            if other!=b:valid&=~mask
        target=masks[b]&valid
        if target.sum()<60 and len(sphere_points)<3:continue
        mesh=meshes[name];vertices=np.asarray(mesh.vertices)
        support_vertices=np.asarray(mesh.convex_hull.vertices) if name in table_supported else None
        supported=name in support_ends and frame<=support_ends[name]
        settings=job.get('object_pixel_refinement',{}).get(name,{})
        if settings.get('rest_upright_mesh_sha256'):
            assert hashlib.sha256((rec/f'mesh_priors/{name}.ply').read_bytes()).hexdigest()==settings['rest_upright_mesh_sha256'],'Stable support prior belongs to a different metric mesh'
        translation_bound=float(settings.get('translation_bound_m',.025))
        rotation_bound=float(settings.get('rotation_bound_rad',.25))
        maximum_translation=float(settings.get('maximum_translation_change_m',.05 if supported else .03))
        maximum_rotation=float(settings.get('maximum_rotation_change_rad',.35))
        assert 0<translation_bound<=.08 and 0<rotation_bound<=.8
        assert 0<maximum_translation<=.12 and 0<maximum_rotation<=1.2
        vertices=vertices[np.linspace(0,len(vertices)-1,min(2400,len(vertices))).astype(int)]
        seed=motion['object_to_world'][frame,b].copy();distance=distance_transform_edt(~masks[b])
        def transform(x):
            out=seed.copy();out[:3,3]+=x[:3];out[:3,:3]=Rotation.from_rotvec(x[3:]).as_matrix()@seed[:3,:3]
            if supported:out[2,3]=.6505-float((support_vertices@out[:3,:3].T)[:,2].min())
            if table is not None and name in table_supported:
                release=max((g['end'] for g in c['grasps'] if c['objects'][g['object']]==name),default=c['frames'])
                settled=supported or frame>=release+4
                if settled and settings.get('rest_upright_local_axis'):
                    up=np.asarray(settings['rest_upright_local_axis'],dtype=float)
                    assert up.shape==(3,) and np.isclose(np.linalg.norm(up),1.,atol=1e-4)
                    alignment=Rotation.align_vectors(np.array([[0.,0.,1.]]),(out[:3,:3]@up)[None])[0]
                    out[:3,:3]=alignment.as_matrix()@out[:3,:3]
                out=constrain_pose(out,collision_vertices.get(name,support_vertices),*table,settled=settled)
            return out
        def score(x,raw=False):
            M=seed if raw else transform(x);points=(vertices@M[:3,:3].T+M[:3,3])@camera[:3,:3].T+camera[:3,3]
            pixels=points[:,:2]/np.maximum(.01,points[:,2:])*[intr['fx']*.5,intr['fy']*.5]+[intr['cx']*.5,intr['cy']*.5]
            uv=np.rint(pixels).astype(int);ok=(points[:,2]>.05)&(uv[:,0]>=0)&(uv[:,0]<512)&(uv[:,1]>=0)&(uv[:,1]<384)
            render=np.zeros((384,512),np.uint8);render[uv[ok,1],uv[ok,0]]=1
            render=cv2.morphologyEx(render,cv2.MORPH_CLOSE,np.ones((3,3),np.uint8))
            contours,_=cv2.findContours(render,cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_SIMPLE);cv2.drawContours(render,contours,-1,1,-1)
            render=render.astype(bool)&valid
            iou=float((render&target).sum()/max(1,(render|target).sum()))
            clipped=np.clip(pixels,[0,0],[511,383]);visible=valid[clipped[:,1].astype(int),clipped[:,0].astype(int)]&ok
            extra=float(np.mean(map_coordinates(distance,clipped[visible,::-1].T,order=1))) if visible.any() else 100.
            return iou,extra
        before,_=score(np.zeros(6),raw=True)
        if seed_motion is not None:
            # Carry forward only improved reconstructed object estimates. Move
            # through camera coordinates so a new SLAM world gauge is allowed.
            proposed=rebase_reconstructed_pose(motion['camera_to_world'][frame],seed_motion['camera_to_world'][frame],seed_motion['object_to_world'][frame,b])
            delta=np.r_[proposed[:3,3]-seed[:3,3],Rotation.from_matrix(proposed[:3,:3]@seed[:3,:3].T).as_rotvec()]
            seeded_score,_=score(delta)
            if seeded_score>before+.005 and np.linalg.norm(delta[:3])<=.05 and np.linalg.norm(delta[3:])<=.5:
                seed=transform(delta);motion['object_to_world'][frame,b]=seed
                records.append({'frame':frame,'object':name,'improved_reconstruction_seed_accepted':True,
                                'visible_iou_before':before,'visible_iou_after':seeded_score})
                before=seeded_score
        if frame not in refit_indexes:continue
        marker_before=sphere_error(seed)
        if before>=.92 and not supported and np.allclose(transform(np.zeros(6)),seed,atol=1e-6) and (marker_before is None or marker_before<=1.):continue
        def loss(x):
            overlap,distance_error=score(x)
            continuity=0.
            if b in previous and previous[b][0]==frame-1:
                delta=x-previous[b][1];continuity=float(settings.get('continuity_weight',.01))*float(np.sum((delta[:3]/.008)**2)+np.sum((delta[3:]/.1)**2))
            sphere_cost=0.
            if sphere_points:
                errors=np.linalg.norm(marker_residual(camera@transform(x),sphere_points,sphere_pixels,intr),axis=1)/3.
                weight=.04 if len(sphere_points)>=3 else .015
                sphere_cost=weight*float(np.mean(np.sqrt(1+errors*errors)-1))
            return 1-overlap+.003*distance_error+.001*float(np.sum((x[:3]/.025)**2)+np.sum((x[3:]/.25)**2))+continuity+sphere_cost
        bounds=[(-translation_bound,translation_bound)]*3+[(-rotation_bound,rotation_bound)]*3
        starts=[np.zeros(6)]
        if settings.get('temporal_multistart') and b in previous:
            starts.append(np.clip(previous[b][1],[-translation_bound]*3+[-rotation_bound]*3,[translation_bound]*3+[rotation_bound]*3))
        candidates=[minimize(loss,start,method='Powell',bounds=bounds,
                     options={'maxiter':int(settings.get('maxiter',12)),'maxfev':int(settings.get('maxfev',240)),'xtol':.0002,'ftol':.0005}) for start in starts]
        fit=min(candidates,key=lambda result:result.fun)
        after,_=score(fit.x)
        changed=transform(fit.x);position_change=float(np.linalg.norm(changed[:3,3]-seed[:3,3]))
        rotation_change=float(Rotation.from_matrix(changed[:3,:3]@seed[:3,:3].T).magnitude())
        marker_after=sphere_error(changed)
        sphere_improved=bool(marker_before is not None and marker_after<marker_before-.5 and marker_after<3. and after>=before-.005
                             and (len(sphere_points)>=3 or (position_change<=.01 and rotation_change<=.1)))
        marker_consistent=marker_before is None or len(sphere_points)<3 or marker_after<=marker_before+.75
        accepted=bool((after>=before+.005 or sphere_improved) and marker_consistent and position_change<=maximum_translation and rotation_change<=maximum_rotation)
        if accepted:motion['object_to_world'][frame,b]=transform(fit.x)
        previous[b]=(frame,fit.x.copy() if accepted else np.zeros(6))
        records.append({'frame':frame,'object':name,'visible_iou_before':before,'visible_iou_after':after,'accepted':accepted,
                        'measured_sphere_count':len(sphere_points),'sphere_error_before_px':marker_before,'sphere_error_after_px':marker_after,
                        'sphere_constraint_improved':sphere_improved,
                        'position_change_m':position_change,'rotation_change_rad':rotation_change,'initial_table_support_constraint':supported})
    if frame%10==0:print('Source-pixel pose refinement',frame,flush=True)
np.savez_compressed(rec/'scene_motion_physical.npz',**motion)
save(marker,{'updated_utc':utc(),'method':__doc__,'frames_refitted':sorted(refit_indexes),'frames_scored_for_seed':indexes,'changes':records,
             'camera_mesh_dimensions_hands_and_source_clock_preserved':True,'policy_states_edited':False,
             'prior_reconstruction_used_as_pixel_checked_seed':job.get('seed_reconstruction_run'),
             'object_refinement_settings':job.get('object_pixel_refinement',{}),
             'finite_table_nonpenetration_during_pose_search':table is not None,
             'physical_collision_geometry_objects':sorted(collision_vertices),
             'locating_spheres_used_as_image_references':bool(sphere_frames is not None),
             'sphere_observation_summary':sphere_summary,
             'evaluation_ground_truth_used':False})
