"""Fit rigid CAD to source images, then export every-frame alignment evidence.

This stage has no robot, policy, physics projection, image warping or time warping.
Original calibrated images, metric geometry and all source timestamps are fixed.
"""
import argparse
import json
import os
from pathlib import Path
import shutil
import subprocess
import tempfile
import time

import cv2
import numpy as np
import trimesh
from scipy.ndimage import distance_transform_edt
from scipy.optimize import minimize
from scipy.spatial.transform import Rotation

from improvement_common import BASE, BATCH, digest, save, utc
from source_mask_rasterizer import MeshSilhouette
from sphere_marker_reference import measure_markers, marker_residual
from source_photometric_reference import MeshPhotometricReference
from metric_surface_localisation import translation_seed
from contour_pose_refinement import propose as contour_pose_proposal,face_flip_proposals,temporal_pose_cost,moving_orientation_proposals
from source_marker_pose_proposals import load_reviews,bootstrap as reviewed_marker_bootstrap,checked_observations
from source_marker_templates import load as load_marker_templates
from source_feature_pose import track as feature_pose_proposal,penalty as feature_pose_penalty

VERSION = 5
ALGORITHM_HASHES={name:digest(BASE/name) for name in ['source_cad_alignment.py','source_cap_masks.py','source_feature_pose.py','source_marker_templates.py','source_marker_pose_proposals.py','source_photometric_reference.py','source_static_joint_fit.py','source_final_scene_audit.py','metric_surface_localisation.py','source_mask_rasterizer.py','source_mask_rasterizer.c','sphere_marker_reference.py','contour_pose_refinement.py']}
LIMITS = dict(minimum_visible_pixels=60, minimum_iou=.90,
              maximum_centroid_error_px=3., maximum_boundary_error_px=3.,
              maximum_sphere_error_px=3.)


def diverse_pose_seeds(candidates, maximum=4):
    """Retain distinct basins from an already quality-sorted source proposal list."""
    selected=[]
    for pose in candidates:
        if not np.isfinite(pose).all():continue
        if all(np.linalg.norm(pose[:3,3]-old[:3,3])>=.005 or
               Rotation.from_matrix(pose[:3,:3]@old[:3,:3].T).magnitude()>=.12
               for old in selected):
            selected.append(pose)
        if len(selected)>=maximum:break
    return selected


def source_centroid_cost(render,target_center_xy):
    """Guide optimization toward the unchanged pixel-centroid gate.

    Zero inside the existing strict tolerance; bounded outside it so this term
    cannot replace silhouette, depth, marker, or temporal evidence.
    """
    if target_center_xy is None:return 0.
    y,x=np.where(render)
    if not len(x):return 1.
    error=2*float(np.linalg.norm(np.array([x.mean(),y.mean()])-target_center_xy))
    excess=max(0.,error/LIMITS['maximum_centroid_error_px']-1.)
    return min(1.,.04*excess**2)


def save_trajectory(path, motion):
    """Keep the previous resumable trajectory if writing is interrupted."""
    temporary=None
    try:
        with tempfile.NamedTemporaryFile(dir=path.parent,prefix='.'+path.name+'.',suffix='.tmp',delete=False) as stream:
            temporary=Path(stream.name)
            np.savez_compressed(stream,**motion)
            stream.flush();os.fsync(stream.fileno())
        os.replace(temporary,path)
    finally:
        if temporary is not None:temporary.unlink(missing_ok=True)


def human_mask(hands, frame):
    mask = np.zeros((768, 1024), np.uint8)
    for side in range(2):
        if not hands['observed'][frame, side]:
            continue
        points = np.rint(hands['uv'][frame, side]).astype(np.int32)
        cv2.fillConvexPoly(mask, cv2.convexHull(points[[0, 1, 5, 9, 13, 17]]), 1)
        for finger in range(5):
            ids = [0] + list(range(1 + 4*finger, 5 + 4*finger))
            for first, last in zip(ids[:-1], ids[1:]):
                cv2.line(mask, tuple(points[first]), tuple(points[last]), 1, 14)
    return mask


def metrics(render, target, valid):
    a, b = render & valid, target & valid
    ay, ax = np.where(a); by, bx = np.where(b)
    iou = float((a & b).sum()/max(1, (a | b).sum()))
    centroid = float(np.linalg.norm([ax.mean()-bx.mean(), ay.mean()-by.mean()])*2) if len(ax) and len(bx) else None
    edge_a = a & ~cv2.erode(a.astype(np.uint8), np.ones((3, 3), np.uint8)).astype(bool)
    edge_b = b & ~cv2.erode(b.astype(np.uint8), np.ones((3, 3), np.uint8)).astype(bool)
    boundary = float((distance_transform_edt(~edge_b)[edge_a].mean()+distance_transform_edt(~edge_a)[edge_b].mean())) if edge_a.any() and edge_b.any() else None
    return dict(iou=iou, centroid_error_px=centroid, boundary_error_px=boundary,
                visible_pixels=int(b.sum()),
                source_centroid_px=[float(bx.mean()*2),float(by.mean()*2)] if len(bx) else None,
                cad_centroid_px=[float(ax.mean()*2),float(ay.mean()*2)] if len(ax) else None)


def qualifies(row):
    """Numerical image agreement is not proof of exact hidden 3D motion."""
    return (row['visible_pixels'] >= LIMITS['minimum_visible_pixels']
            and row['iou'] >= LIMITS['minimum_iou']
            and row['centroid_error_px'] is not None
            and row['centroid_error_px'] <= LIMITS['maximum_centroid_error_px']
            and row['boundary_error_px'] is not None
            and row['boundary_error_px'] <= LIMITS['maximum_boundary_error_px']
            and (row.get('sphere_error_px') is None or row['sphere_error_px'] <= LIMITS['maximum_sphere_error_px']))


def choose_qualified_source_pose(candidates,best,best_quality,measure_pose,quality):
    """Minimise the existing objective within the unchanged source gates.

    A better soft depth/texture score must not erase an available source-valid
    pose. If none qualifies, retain the failed pose and its failure evidence.
    """
    original=measure_pose(best);seen=set();feasible=[]
    for candidate in [best,*candidates]:
        key=np.asarray(candidate).tobytes()
        if key in seen:continue
        seen.add(key);row=measure_pose(candidate)
        if not qualifies(row):continue
        value=quality(candidate)
        if np.isfinite(value):feasible.append((value,candidate,row))
    info=dict(unique_candidates_checked=len(seen),qualified_candidates=len(feasible),
              replaced_unqualified_best=bool(feasible) and not qualifies(original))
    if feasible:
        value,pose,row=min(feasible,key=lambda item:item[0]);return pose,value,row,info
    return best,best_quality,original,info


def verify_current_source_masks(source,cfg):
    """Refuse CAD fitting after source prompt edits invalidated propagation."""
    if not cfg.get('validated_video_pipeline'):return None
    rec=Path(source)/'reconstruction';prompts=rec/'sam2_prompts.json'
    propagation=rec/'masks/propagation-report.json';review=rec/'source-prompt-review.json'
    if not prompts.is_file() or not propagation.is_file():
        raise RuntimeError('Validated CAD requires source prompts and mask propagation provenance')
    active=digest(prompts);report=json.loads(propagation.read_text());prompt_data=json.loads(prompts.read_text())
    reviewed=json.loads(review.read_text()) if review.is_file() else prompt_data
    semantic_keys=['prompts','exclusive_visible_instances','cross_instance_negatives','source_frames_sha256']
    if report.get('prompts_sha256')!=active or any(reviewed.get(k)!=prompt_data.get(k) for k in semantic_keys):
        raise RuntimeError('Source mask prompts/review changed after propagation; rebuild masks and dependent object poses in a fresh frontend revision before CAD fitting')
    if prompt_data.get('exclusive_visible_instances') and (
        report.get('visible_instance_ownership')!='highest measured SAM logit at overlapping foreground pixels'
        or report.get('overlapping_foreground_pixels_after_ownership')!=0
        or report.get('ownership_adds_no_source_foreground') is not True):
        raise RuntimeError('Requested exclusive visible source instances lack verified propagation ownership metadata')
    if prompt_data.get('cross_instance_negatives') and report.get('cross_instance_negative_conditioning_requested') is not True:
        raise RuntimeError('Source mask propagation did not apply requested cross-instance conditioning')
    from source_cap_masks import verify as verify_source_caps
    cap_binding=verify_source_caps(source,active)
    binding=dict(source_prompts_sha256=active,propagation_prompts_sha256=report['prompts_sha256'],
                 source_review_sha256=digest(review) if review.is_file() else None)
    if cap_binding:binding['source_cap_refinement']=cap_binding
    if cap_binding:
        cap_masks=json.loads((Path(source)/'research/source-cap-mask-refinement.json').read_text())['masks_sha256']
        depth_path=Path(source)/'reconstruction/stereo-depth-alignment.json'
        depth=json.loads(depth_path.read_text()) if depth_path.exists() else {}
        if depth.get('source_cap_masks_sha256')!=cap_masks:
            raise RuntimeError('Source cap masks changed after metric depth; rebuild dependent object poses in a fresh frontend revision before CAD fitting')
    return binding


def object_visibility(masks,depths,body):
    """Resolve overlapping object labels by front-surface depth, not mutual erasure."""
    visible=np.ones_like(masks[body],dtype=bool)
    own=depths[body]
    for other in range(len(masks)):
        if other==body:continue
        front=(depths[other]>.05)&((own<=.05)|(depths[other]<own-.002))
        visible &= ~(masks[other]&front)
    return visible


def prepare_budget_mesh(mesh,face_budget):
    """Simplify geometric connectivity rather than disconnected UV islands."""
    if not isinstance(face_budget,int) or face_budget<4:
        raise ValueError('A mesh face budget must be an integer of at least four')
    result=mesh.copy()
    if len(result.faces)>face_budget:
        # Scans duplicate positions at texture and normal seams. Leaving these
        # artificial boundaries disconnected lets QEM remove surface patches.
        # Welding equal positions changes no source triangle or metric frame.
        result.merge_vertices(merge_tex=True,merge_norm=True)
        result=result.simplify_quadric_decimation(face_count=face_budget)
    if len(result.faces)>face_budget:
        raise RuntimeError('Simplified mesh exceeded the submission face budget')
    return result


def source_feature_reference_available(name,frame,records):
    """Only extend white-pot image tracking from a checked previous surface."""
    if frame<=0:return False
    if name in {'wooden_piece_1','wooden_piece_2'}:return True
    if name!='white_pot':return False
    for row in reversed(records):
        if row['frame']==frame-1 and row['object']==name:
            return row.get('local_overlap_pass') is True
    return False


def initialize(source, root):
    root.mkdir(parents=True, exist_ok=True)
    for folder in ['reconstruction', 'research', 'videos', 'evaluation', 'logs']:
        (root/folder).mkdir(exist_ok=True)
    cfg = json.loads((source/'episode-config.json').read_text())
    validation=BATCH/'focused_episodes_20261028/pipeline_validation_20261029/status.json'
    if cfg.get('dataset_split')!='public' and validation.exists():
        admitted=json.loads(validation.read_text())
        if admitted.get('evaluation_training_admitted') and cfg['episode'] in admitted.get('evaluation_scope',[]):
            cfg.update(validated_video_pipeline=True,source_only_reconstruction=True,submission_mesh_face_budget=3600,
                       public_validation_manifest=str(validation),time_advance_frames=0.,
                       photometric_refinement=True,static_joint_refinement=True,full_sequence_stereo_required=True)
    cfg.update(run=str(root), source_reconstruction_run=str(source),
               source_cad_only=True, policy_training_started=False)
    probe=cv2.VideoCapture(cfg['source'])
    fps=probe.get(cv2.CAP_PROP_FPS);frames=int(probe.get(cv2.CAP_PROP_FRAME_COUNT));probe.release()
    assert frames==cfg['frames'] and fps>0,'Source frame count or clock is unavailable'
    cfg['fps']=fps
    save(root/'episode-config.json', cfg)
    rec = root/'reconstruction'
    # Link immutable image/depth inputs; keep every mutable output isolated.
    for name in ['frames', 'masks', 'depth', 'hand_measurements.npz',
                 'camera_intrinsics.json', 'calibrated.mp4', 'camera-frame-calibration.json',
                 'table-camera-calibration.json','depth_stereo','stereo_valid']:
        src = source/'reconstruction'/name; dst = rec/name
        if cfg.get('validated_video_pipeline') and name=='depth_stereo' and (source/'reconstruction/foundation-stereo-sequence.json').is_file():
            src=source/'reconstruction/depth_foundation_stereo'
            cfg['source_depth_variant']='FoundationStereo with left/right consistency'
        if src.exists() and not dst.exists():dst.symlink_to(src)
    assets = rec/'mesh_priors'; assets.mkdir(exist_ok=True)
    geometry = {}
    for name in cfg['objects']:
        # Preserve the metric scan frame. Missing spoon geometry is explicitly
        # labelled a reconstruction estimate, never a released CAD asset.
        folder=BASE/f'datasets/video_to_data_challenge/track_3/public/mesh/{name}'
        public=next((p for p in [folder/f'{name}_visual.glb',folder/f'{name}.glb'] if p.exists()),folder/f'{name}_visual.glb')
        src = public if public.exists() else source/f'reconstruction/mesh_priors/{name}.ply'
        dst = assets/f'{name}.ply'
        if not dst.exists():
            if public.exists():
                mesh=trimesh.load(src,force='scene',process=False).to_geometry()
                budget=cfg.get('submission_mesh_face_budget')
                if budget and len(mesh.faces)>budget:
                    mesh=prepare_budget_mesh(mesh,budget)
                mesh.export(dst)
            else:shutil.copy2(src, dst)
        mesh = trimesh.load(dst, process=False)
        geometry[name] = dict(path=str(dst), sha256=digest(dst),
                              vertices=len(mesh.vertices), faces=len(mesh.faces),
                              extents_m=mesh.extents.tolist(),
                              source='video reconstruction estimate; thickness remains a prior' if name=='wooden_spoon' else 'released metric CAD scan fitted to this episode',
                              source_mesh=str(src), source_mesh_sha256=digest(src))
        if not (assets/f'{name}.glb').exists():
            if public.exists() and not cfg.get('submission_mesh_face_budget'):shutil.copy2(public,assets/f'{name}.glb')
            else:mesh.export(assets/f'{name}.glb')
    raw=source/'reconstruction/scene_motion.npz'
    src=raw if cfg.get('source_only_reconstruction') and raw.exists() else source/'reconstruction/scene_motion_physical.npz'
    initial = rec/'source-initial-motion.npz'
    if not initial.exists():shutil.copy2(src, initial)
    save(root/'research/cad-geometry.json', dict(updated_utc=utc(), objects=geometry,
         source_inputs_modified=False, geometry_scale_fitted_per_frame=False,
         missing_official_assets=[n for n in cfg['objects'] if n=='wooden_spoon']))
    save(root/'episode-config.json',cfg)
    return cfg


def preview(root,cfg,motion,renderers,frame):
    rec=root/'reconstruction'
    image=cv2.imread(str(rec/f"frames/{frame:06d}.{cfg.get('frame_cache_format','png')}"));canvas=image.copy()
    C=np.linalg.inv(motion['camera_to_world'][frame]);colors=[(30,220,60),(230,70,220),(255,180,30)]
    for b,name in enumerate(cfg['objects']):
        mask=renderers[name].render(C@motion['object_to_world'][frame,b]).astype(np.uint8)
        contours,_=cv2.findContours(mask,cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_SIMPLE)
        cv2.drawContours(canvas,[x*2 for x in contours],-1,colors[b%len(colors)],2)
    cv2.putText(canvas,f'CAD / SOURCE diagnostic | frame {frame} | not a policy rollout',(12,30),0,.6,(255,255,255),2)
    cv2.imwrite(str(root/'research/cad-live-preview.jpg'),np.concatenate([cv2.resize(image,(768,576)),cv2.resize(canvas,(768,576))],axis=1))


def fit(source, root, maxfev=100, repair_only=False):
    source_cfg=json.loads((source/'episode-config.json').read_text())
    source_mask_binding=verify_current_source_masks(source,source_cfg)
    validation=BATCH/'focused_episodes_20261028/pipeline_validation_20261029/status.json'
    admitted=json.loads(validation.read_text()) if validation.exists() else {}
    needs_stereo=source_cfg.get('full_sequence_stereo_required') or (source_cfg.get('dataset_split')!='public' and admitted.get('evaluation_training_admitted') and source_cfg['episode'] in admitted.get('evaluation_scope',[]))
    if needs_stereo:
        from foundation_stereo_sequence import run as stereo
        root.mkdir(exist_ok=True,parents=True);(root/'research').mkdir(exist_ok=True)
        save(root/'research/cad-progress.json',dict(updated_utc=utc(),episode=source_cfg['episode'],pid=os.getpid(),
             phase='FoundationStereo full-sequence metric depth before CAD fitting',frame=0,total_frames=source_cfg['frames'],
             object_frames_processed=0,total_object_frames=source_cfg['frames']*len(source_cfg['objects']),policy_training_started=False))
        stereo(source,progress_root=root)
    cfg=initialize(source, root); rec=root/'reconstruction'; count=cfg['frames']
    if source_mask_binding:save(root/'research/source-mask-provenance.json',source_mask_binding)
    initial=np.load(rec/'source-initial-motion.npz'); motion={k:initial[k].copy() for k in initial.files}
    checkpoint=rec/'cad-source-trajectory.npz'; records_path=root/'research/cad-frame-metrics.json'
    records=json.loads(records_path.read_text()) if records_path.exists() else []
    if checkpoint.exists():motion['object_to_world']=np.load(checkpoint)['object_to_world']
    intr=json.loads((rec/'camera_intrinsics.json').read_text());hands=np.load(rec/'hand_measurements.npz')
    marker_reviews,marker_review_binding=load_reviews(cfg,rec)
    reviewed_marker_objects={row['object'] for row in marker_reviews}
    save(root/'research/source-marker-review.json',dict(binding=marker_review_binding,reviewed_objects=sorted(reviewed_marker_objects),
         source_image_points_only=True,ground_truth_used=False,does_not_prove_episode_completion=True))
    if repair_only:
        # Refit failed, observable frames; retain already good measurements and
        # do not invent data for frames where the object is hidden.
        records=[r for r in records if r['local_overlap_pass'] or r['visibility']!='visible']
    done={(r['frame'],r['object']) for r in records}
    assert len(motion['object_to_world'])==count
    templates=load_marker_templates(BATCH/'focused_episodes_20261028/huggingface_audit_20261028/marker-templates.json',
         BATCH/'focused_episodes_20261028/source-marker-feature-review.json',root/'research/source-marker-template-review.json')
    if 'wooden_spoon' in cfg['objects']:
        from spoon_marker_geometry import recover
        spoon=recover(source,cfg,motion,intr,root)
        if spoon['markers']:
            templates['wooden_spoon']=spoon
            geometry_marker=root/'research/spoon-sphere-cad.json'
            if not geometry_marker.exists():
                mesh=trimesh.load(rec/'mesh_priors/wooden_spoon.ply',process=False)
                parts=[mesh]
                for item in spoon['markers']:
                    ball=trimesh.creation.icosphere(subdivisions=2,radius=item['radius_m'])
                    ball.apply_translation(item['center_m']);parts.append(ball)
                rebuilt=trimesh.util.concatenate(parts)
                rebuilt.export(rec/'mesh_priors/wooden_spoon.ply');rebuilt.export(rec/'mesh_priors/wooden_spoon.glb')
                save(geometry_marker,dict(updated_utc=utc(),sphere_count=len(spoon['markers']),
                     method='Add repeatedly observed metric stereo sphere candidates to the episode spoon CAD. Source overlay validation remains mandatory.',
                     mesh_sha256=digest(rec/'mesh_priors/wooden_spoon.ply')))
    renderers={};photometric={}
    for name in cfg['objects']:
        mesh=trimesh.load(rec/f'mesh_priors/{name}.ply',process=False)
        vertices,inverse=np.unique(mesh.vertices,axis=0,return_inverse=True)
        faces=inverse[mesh.faces]
        renderers[name]=MeshSilhouette(vertices,faces,intr)
        if cfg.get('photometric_refinement'):
            folder=BASE/f'datasets/video_to_data_challenge/track_3/public/mesh/{name}'
            texture=next((p for p in [folder/f'{name}_visual.glb',folder/f'{name}.glb'] if p.exists()),None)
            photometric[name]=MeshPhotometricReference.from_mesh(texture) if texture else None
    save(root/'research/photometric-reference.json',dict(enabled=bool(cfg.get('photometric_refinement')),
         method='Released scan texture samples projected into each source RGB frame; front-surface depth and hand/object occlusion checks; per-view exposure normalization',
         objects={name:dict(available=reference is not None,samples=len(reference.points) if reference else 0) for name,reference in photometric.items()},ground_truth_used=False))
    def progress(stage, frame):
        save(root/'research/cad-progress.json',dict(updated_utc=utc(),episode=cfg['episode'],
             pid=os.getpid(),phase=stage,frame=frame,total_frames=count,
             object_frames_processed=len(records),total_object_frames=count*len(cfg['objects']),
             policy_training_started=False))
    for frame in range(count):
        image=cv2.imread(str(rec/f"frames/{frame:06d}.{cfg.get('frame_cache_format','png')}"));assert image is not None
        image_rgb=cv2.resize(cv2.cvtColor(image,cv2.COLOR_BGR2RGB).astype(np.float32)/255.,(512,384))
        occlusion=human_mask(hands,frame); camera=np.linalg.inv(motion['camera_to_world'][frame])
        masks=[cv2.imread(str(rec/f'masks/{b+1}/{frame:06d}.png'),0)>0 for b in range(len(cfg['objects']))]
        small_masks=[cv2.resize(m.astype(np.uint8),(512,384),interpolation=cv2.INTER_NEAREST)>0 for m in masks]
        depths=([renderers[name].depth(camera@motion['object_to_world'][frame,b]) for b,name in enumerate(cfg['objects'])]
                if cfg.get('validated_video_pipeline') else None)
        stereo=None
        if cfg.get('validated_video_pipeline') and (rec/f'depth_stereo/{frame:06d}.png').exists():
            encoded=cv2.imread(str(rec/f'depth_stereo/{frame:06d}.png'),-1)
            stereo=cv2.resize(np.where(encoded>0,65535/np.maximum(encoded.astype(float),1)-1,0),(512,384),interpolation=cv2.INTER_NEAREST)
        for body,name in enumerate(cfg['objects']):
            if (frame,name) in done:continue
            renderer=renderers[name];target=cv2.resize(masks[body].astype(np.uint8),(512,384),interpolation=cv2.INTER_NEAREST)>0
            valid=cv2.resize(occlusion,(512,384),interpolation=cv2.INTER_NEAREST)==0
            if depths is not None:valid &= object_visibility(small_masks,depths,body)
            else:
                for other in range(len(masks)):
                    if other!=body:valid &= ~small_masks[other]
            template=templates.get(name); seed=camera@motion['object_to_world'][frame,body]
            metric_seed=translation_seed(renderer,seed,intr,stereo,target&valid,visibility=valid) if stereo is not None else None
            feature=None;independent_markers=False
            if cfg.get('source_feature_pose_enabled') and source_feature_reference_available(name,frame,records):
                previous_image=cv2.imread(str(rec/f"frames/{frame-1:06d}.{cfg.get('frame_cache_format','png')}"))
                previous_mask=cv2.imread(str(rec/f'masks/{body+1}/{frame-1:06d}.png'),0)>0
                previous_pose=np.linalg.inv(motion['camera_to_world'][frame-1])@motion['object_to_world'][frame-1,body]
                initial=camera@motion['object_to_world'][frame-1,body]
                current_occlusion=cv2.resize((~valid).astype(np.uint8),masks[body].shape[::-1],interpolation=cv2.INTER_NEAREST)>0
                feature=feature_pose_proposal(previous_image,image,renderer,previous_pose,intr,previous_mask,masks[body],
                    human_mask(hands,frame-1),current_occlusion,initial)
            # Establish a source silhouette/depth proposal before associating
            # locating spheres. A shifted depth-centroid seed must not lock a
            # nearby, unrelated image circle onto a CAD marker identity.
            observations=[];good=[];xyz=[];uv=[]
            target_visible=target & valid; distance=distance_transform_edt(~target_visible)
            target_y,target_x=np.where(target_visible)
            target_center_xy=np.array([target_x.mean(),target_y.mean()]) if len(target_x) else None
            def sphere_error(M):
                return float(np.median(np.linalg.norm(marker_residual(M,xyz,uv,intr),axis=1))) if xyz else None
            def quality(M):
                render=renderer.render(M)&valid
                iou=float((render&target_visible).sum()/max(1,(render|target_visible).sum()))
                outside=float(distance[render].mean()) if render.any() else 100.
                sphere=sphere_error(M)
                depth_loss=0.;photo_loss=0.;surface=None
                if stereo is not None:
                    surface=renderer.depth(M);measured=target_visible&render&(stereo>.05)&(surface>.05)
                    if measured.sum()>=20:
                        error=np.clip(np.abs(surface[measured]-stereo[measured]),0,.10)
                        depth_loss=.04*float(np.median(np.sqrt(1+(error/.01)**2)-1))
                photo=photometric.get(name)
                if photo is not None:
                    if surface is None:surface=renderer.depth(M)
                    photo_loss=.03*photo.residual(M,image_rgb,intr,surface,target_visible)[0]
                center_loss=source_centroid_cost(render,target_center_xy) if cfg.get('validated_video_pipeline') else 0.
                return (1-iou+.005*outside+depth_loss+photo_loss+center_loss
                    +(.06*float(np.sqrt(1+(sphere/3)**2)-1) if sphere is not None else 0.)
                    +feature_pose_penalty(M,feature,intr,independent_markers))
            def selection_quality(M):
                temporal=temporal_pose_cost(motion['camera_to_world'][frame]@M,motion['object_to_world'][frame-1,body],motion['object_to_world'][frame-2,body]) if frame>1 else 0.
                return quality(M)+temporal
            before_render=renderer.render(seed);before=metrics(before_render,target,valid)
            best=seed;best_quality=selection_quality(seed)
            candidates=[seed]
            if metric_seed is not None:candidates.append(metric_seed)
            if feature is not None:candidates.append(feature['pose'])
            if cfg.get('validated_video_pipeline') and frame and name in {'white_pot_lid','wooden_piece_1','wooden_piece_2','mini_sweeper'}:
                orientations=moving_orientation_proposals(renderer,seed,motion['camera_to_world'][frame],
                    motion['object_to_world'][frame-1,body],motion['object_to_world'][frame-2,body] if frame>1 else None)
                candidates.extend(orientations)
                if stereo is not None:
                    for proposal in orientations:
                        localised=translation_seed(renderer,proposal,intr,stereo,target&valid,visibility=valid)
                        if localised is not None:candidates.append(localised)
            if cfg.get('planar_face_flip_proposals_enabled') and name in {'white_pot_lid','wooden_piece_1','wooden_piece_2'}:
                for M in face_flip_proposals(renderer,seed):
                    candidates.append(M)
                    if stereo is not None:
                        translated=translation_seed(renderer,M,intr,stereo,target&valid,visibility=valid)
                        if translated is not None:candidates.append(translated)
            if frame:
                # Offer the previous pose before contour refinement so it can
                # follow motion measured in this frame, rather than competing
                # only as an unchanged old pose. Current source checks still
                # decide whether either candidate is acceptable.
                prior=camera@motion['object_to_world'][frame-1,body]
                candidates.append(prior)
            if cfg.get('contour_pose_refinement_enabled') and not qualifies(before):
                for start in list(candidates):
                    refined=contour_pose_proposal(renderer,start,target,valid,intr,stereo)
                    if refined is not None:candidates.append(refined)
            if cfg.get('validated_video_pipeline') and (frame==0 or before['iou']<.8):
                # Escape a wrong global orientation using this frame's image
                # and measured depth, without using public/evaluation poses.
                world=motion['camera_to_world'][frame]@seed
                for yaw in np.linspace(-np.pi,np.pi,12,endpoint=False):
                    M=world.copy();M[:3,:3]=Rotation.from_rotvec([0.,0.,yaw]).as_matrix()@world[:3,:3]
                    M=camera@M;candidates.append(M)
                    if stereo is not None:
                        localised=translation_seed(renderer,M,intr,stereo,target&valid,visibility=valid)
                        if localised is not None:candidates.append(localised)
                _,axes=np.linalg.eigh(np.cov(renderer.vertices.T));axis=axes[:,-1]
                for angle in np.linspace(-np.pi/4,np.pi/4,7):
                    M=seed.copy();M[:3,:3]=seed[:3,:3]@Rotation.from_rotvec(axis*angle).as_matrix()
                    candidates.append(M)
            if template:
                marker_seed=min(candidates,key=quality)
                reviewed=next((row for row in marker_reviews if row['object']==name and row['frame']==frame),None)
                if reviewed:
                    proposals=reviewed_marker_bootstrap(image,template,renderer,intr,masks[body],occlusion,target,valid,reviewed['points'],metrics,qualifies)
                    candidates.extend(row['pose'] for row in proposals)
                strong=[]
                if name in reviewed_marker_objects or feature is not None:
                    for candidate in candidates:
                        checked=checked_observations(image,template,candidate,intr,masks[body],occlusion,renderer,target,valid,metrics,qualifies)
                        if checked:strong.append((quality(candidate),candidate,checked[0]))
                if strong:
                    independent_markers=True
                    _,marker_seed,observations=min(strong,key=lambda row:quality(row[1]))
                else:
                    observations=measure_markers(image,template,marker_seed,intr,masks[body],occlusion,surface_depth=renderer.depth(marker_seed))
                good=[x for x in observations if x['observed'] and x['measurement']['cost']<4.5]
                xyz=[template['markers'][x['marker_id']]['center_m'] for x in good]
                uv=[x['measurement']['xy'] for x in good]
                best_quality=selection_quality(seed)
            if len(xyz)>=3:
                try:
                    k=np.array([[intr['fx'],0,intr['cx']],[0,intr['fy'],intr['cy']],[0,0,1.]])
                    _,rvecs,tvecs,_=cv2.solvePnPGeneric(np.array(xyz,dtype=float),np.array(uv,dtype=float),k,None,flags=cv2.SOLVEPNP_SQPNP)
                    for rv,tv in zip(rvecs,tvecs):
                        M=np.eye(4);M[:3,:3]=cv2.Rodrigues(rv)[0];M[:3,3]=tv.ravel()
                        if M[2,3]>.05 and np.linalg.norm(M[:3,3]-seed[:3,3])<.10:candidates.append(M)
                except cv2.error:pass
            candidates.sort(key=selection_quality)
            optimised_candidates=[]
            if target_visible.sum()>=60:
                # Neighboring proposals often describe the same wrong minimum.
                # Spend the existing total evaluation budget across distinct
                # orientations/translations rather than optimizing duplicates.
                starts=diverse_pose_seeds(candidates)
                per_start_budget=max(1,2*maxfev//max(1,len(starts)))
                for base in starts:
                    def pose(x):
                        M=base.copy();M[:3,3]+=x[:3];M[:3,:3]=Rotation.from_rotvec(x[3:]).as_matrix()@base[:3,:3];return M
                    def loss(x):
                        return selection_quality(pose(x))+.0002*float(np.sum((x[:3]/.05)**2)+np.sum((x[3:]/.5)**2))
                    result=minimize(loss,np.zeros(6),method='Powell',bounds=[(-.06,.06)]*3+[(-.55,.55)]*3,
                                    options=dict(maxfev=per_start_budget,maxiter=8,xtol=.0002,ftol=.0002))
                    for M in [base,pose(result.x)]:
                        optimised_candidates.append(M)
                        q=selection_quality(M)
                        if q<best_quality:best,best_quality=M,q
                    if metrics(renderer.render(best),target,valid)['iou']>.95 and (sphere_error(best) is None or sphere_error(best)<2.):break
            after=metrics(renderer.render(best),target,valid);after['sphere_error_px']=sphere_error(best)
            source_selection=None
            if cfg.get('validated_video_pipeline'):
                def measure_pose(M):
                    row=metrics(renderer.render(M),target,valid);row['sphere_error_px']=sphere_error(M);return row
                best,best_quality,after,source_selection=choose_qualified_source_pose(
                    candidates+optimised_candidates,best,best_quality,measure_pose,selection_quality)
            # A marker assignment alone must not degrade the body silhouette.
            if after['iou']<before['iou']-.005 and not (cfg.get('validated_video_pipeline') and qualifies(after)):
                best=seed;after=before.copy();after['sphere_error_px']=sphere_error(seed)
            motion['object_to_world'][frame,body]=motion['camera_to_world'][frame]@best
            if xyz:
                projected=marker_residual(best,xyz,np.zeros((len(xyz),2)),intr)
                for item,fitted in zip(good,projected):item['fitted_xy']=fitted.tolist()
            step=float(np.linalg.norm(motion['object_to_world'][frame,body,:3,3]-motion['object_to_world'][frame-1,body,:3,3])) if frame else None
            angular=float(Rotation.from_matrix(motion['object_to_world'][frame-1,body,:3,:3].T@motion['object_to_world'][frame,body,:3,:3]).magnitude()) if frame else None
            row=dict(frame=frame,time_s=frame/float(cfg['fps']),object=name,before=before,**after,
                     sphere_observations=good,measured_sphere_count=len(good),
                     marker_template_available=template is not None,
                     translation_step_m=step,rotation_step_rad=angular,
                     local_overlap_pass=qualifies(after),
                     visibility='unverified_occluded' if after['visible_pixels']<60 else 'visible')
            if source_selection is not None:row['strict_source_candidate_selection']=source_selection
            if feature is not None:
                row['source_feature_tracking']=dict(previous_frame=frame-1,current_frame=frame,
                    tracked_count=feature['tracked_count'],inlier_count=feature['inlier_count'],
                    proposal_reprojection_error_px=feature['median_reprojection_error_px'],
                    median_forward_backward_error_px=feature['median_forward_backward_error_px'],
                    independent_current_markers_override=independent_markers,
                    object_points=feature['object_points'].tolist(),image_points=feature['image_points'].tolist(),
                    source_image_correspondences=True,ground_truth_used=False)
            records.append(row)
        save_trajectory(checkpoint,motion);save(records_path,records)
        progress('CAD pose fitting against source pixels and locating spheres',frame+1)
        if frame%10==0:
            preview(root,cfg,motion,renderers,frame)
            print('CAD source fit',cfg['episode'],frame+1,'/',count,flush=True)
    progress('Rendering CAD over the original source frames',count)
    if cfg.get('static_joint_refinement'):
        from source_static_joint_fit import refine
        progress('Joint source-image fit of static segments',count)
        static=refine(root,cfg,motion,renderers,photometric,records,hands,intr,human_mask,object_visibility,metrics,qualifies,templates)
        save(root/'research/static-joint-fit.json',static)
        for row in records:
            frame=row['frame'];body=cfg['objects'].index(row['object'])
            row['translation_step_m']=float(np.linalg.norm(motion['object_to_world'][frame,body,:3,3]-motion['object_to_world'][frame-1,body,:3,3])) if frame else None
            row['rotation_step_rad']=float(Rotation.from_matrix(motion['object_to_world'][frame-1,body,:3,:3].T@motion['object_to_world'][frame,body,:3,:3]).magnitude()) if frame else None
        save_trajectory(checkpoint,motion);save(records_path,records)
    from source_final_scene_audit import audit as audit_final_scene
    final_audit=audit_final_scene(root,cfg,motion,renderers,records,hands,intr,templates,
                                human_mask,object_visibility,metrics,qualifies,progress)
    records.sort(key=lambda row:(row['frame'],row['object']))
    save(records_path,records)
    final_audit.update(updated_utc=utc(),trajectory_sha256=digest(checkpoint),frames_report_sha256=digest(records_path))
    save(root/'research/cad-final-scene-audit.json',final_audit)
    render_evidence(root,cfg,motion,renderers,records,hands)
    source_video=Path(cfg['source'])
    records.sort(key=lambda x:(x['frame'],x['object']));save(records_path,records)
    summary=dict(updated_utc=utc(),version=VERSION,episode=cfg['episode'],run=str(root),source_run=str(source),
                 algorithm_files_sha256=ALGORITHM_HASHES,
                 final_scene_occlusion_rechecked=final_audit['all_object_frames_rechecked'],
                 final_scene_audit=str(root/'research/cad-final-scene-audit.json'),
                 frames_report_sha256=digest(records_path),
                 status='needs_alignment_improvement',source_frames=count,object_frames_checked=len(records),
                 thresholds=LIMITS,all_source_frames_checked=len(records)==count*len(cfg['objects']),
                 local_overlap_failed_object_frames=sum(not x['local_overlap_pass'] for x in records),
                 unobservable_object_frames=sum(x['visibility']!='visible' for x in records),
                 exact_human_motion_verified=False,episode_done=False,policy_training_started=False,
                 source_video_sha256=digest(source_video),trajectory_sha256=digest(checkpoint),
                 geometry_sha256={n:digest(rec/f'mesh_priors/{n}.ply') for n in cfg['objects']},
                 masks_are_estimates_not_ground_truth=True,source_inputs_modified=False,
                 source_image_preprocessing='Fixed factory undistortion and 1024x768 source frame extraction; no object-dependent image warp',
                 source_time_warping_used=False,physical_projection_applied=False,
                 missing_marker_templates=[n for n in cfg['objects'] if n not in templates],
                 objects={n:dict(mean_iou=float(np.mean([x['iou'] for x in records if x['object']==n])),
                     before_mean_iou=float(np.mean([x['before']['iou'] for x in records if x['object']==n])),
                     failed_frames=[x['frame'] for x in records if x['object']==n and not x['local_overlap_pass']],
                     sphere_measurements=sum(x['measured_sphere_count'] for x in records if x['object']==n)) for n in cfg['objects']},
                 overlay=str(root/'videos/cad-source-overlay.mp4'),frames_report=str(records_path),
                 limitation='Visible image alignment cannot prove exact hidden 3D motion. Occluded frames and unavailable spoon sphere geometry remain explicitly unverified.')
    if summary['local_overlap_failed_object_frames']==0:summary['status']='local_image_overlap_pass_exact_motion_unverified'
    save(root/'research/cad-alignment-report.json',summary);progress('CAD alignment audit completed; inspect failed frames',count)
    return summary


def render_evidence(root,cfg,motion,renderers,records,hands):
    rec=root/'reconstruction';fps=float(cfg['fps']);temporary=root/'videos/cad-source-overlay.tmp.mp4'
    writer=cv2.VideoWriter(str(temporary),cv2.VideoWriter_fourcc(*'mp4v'),fps,(1536,576));assert writer.isOpened()
    lookup={(x['frame'],x['object']):x for x in records};colors=[(30,220,60),(230,70,220),(255,180,30)]
    selected=set(np.linspace(0,cfg['frames']-1,12).round().astype(int));panels=[]
    for frame in range(cfg['frames']):
        image=cv2.imread(str(rec/f"frames/{frame:06d}.{cfg.get('frame_cache_format','png')}"));overlay=image.copy()
        camera=np.linalg.inv(motion['camera_to_world'][frame]);occ=human_mask(hands,frame)>0
        for b,name in enumerate(cfg['objects']):
            row=lookup[frame,name];silhouette=renderers[name].render(camera@motion['object_to_world'][frame,b])
            silhouette=cv2.resize(silhouette.astype(np.uint8),(1024,768),interpolation=cv2.INTER_NEAREST)>0
            visible=silhouette&~occ;color=colors[b%len(colors)]
            overlay[visible]=(overlay[visible]*.65+np.array(color)*.35).astype(np.uint8)
            contours,_=cv2.findContours(silhouette.astype(np.uint8),cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_SIMPLE)
            cv2.drawContours(overlay,contours,-1,color,2)
            for item in row['sphere_observations']:
                measured=tuple(np.rint(item['measurement']['xy']).astype(int))
                cv2.circle(overlay,measured,5,(0,255,255),1)
                if 'fitted_xy' in item:
                    fitted=tuple(np.rint(item['fitted_xy']).astype(int))
                    cv2.drawMarker(overlay,fitted,(0,80,255),cv2.MARKER_CROSS,7,1)
                    cv2.line(overlay,measured,fitted,(0,180,255),1)
            cv2.putText(overlay,f'{name} IoU={row["iou"]:.3f} '+('PASS' if row['local_overlap_pass'] else 'UNVERIFIED / FAIL'),(12,72+b*25),0,.55,color,2)
        for canvas,label in [(image,'HUMAN SOURCE | factory-undistorted'),(overlay,'RIGID CAD OVERLAY | yellow: measured spheres')]:
            cv2.rectangle(canvas,(0,0),(1024,44),(15,15,15),-1)
            cv2.putText(canvas,f'{label} | frame {frame} | {frame/fps:.2f}s',(10,28),0,.55,(255,255,255),1)
        pair=np.concatenate([cv2.resize(image,(768,576)),cv2.resize(overlay,(768,576))],axis=1)
        writer.write(pair)
        if frame in selected:panels.append(cv2.resize(pair,(768,288)))
    writer.release()
    subprocess.run(['ffmpeg','-hide_banner','-loglevel','error','-y','-i',str(temporary),'-c:v','libx264','-threads','2','-preset','fast','-crf','21','-pix_fmt','yuv420p','-movflags','+faststart',str(root/'videos/cad-source-overlay.mp4')],check=True)
    temporary.unlink()
    for i in range(0,len(panels),4):cv2.imwrite(str(root/f'research/cad-contact-sheet-{i//4:02d}.jpg'),np.concatenate(panels[i:i+4]))


if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('--source',type=Path,required=True);parser.add_argument('--run',type=Path,required=True);parser.add_argument('--maxfev',type=int,default=100)
    parser.add_argument('--repair-only',action='store_true')
    args=parser.parse_args();fit(args.source,args.run,args.maxfev,args.repair_only)
