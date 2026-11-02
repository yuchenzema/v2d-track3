"""Recheck every source overlay against all objects' final fitted poses."""
import cv2
import numpy as np
from scipy.spatial.transform import Rotation
from sphere_marker_reference import measure_markers,marker_residual


def audit(root,cfg,motion,renderers,records,hands,intrinsics,templates,
          hand_mask,visibility,measure,qualifies,progress=None):
    rec=root/'reconstruction';lookup={(row['frame'],row['object']):row for row in records}
    changed=0
    for frame in range(cfg['frames']):
        camera=np.linalg.inv(motion['camera_to_world'][frame])
        poses=[camera@motion['object_to_world'][frame,body] for body in range(len(cfg['objects']))]
        full_masks=[cv2.imread(str(rec/f'masks/{body+1}/{frame:06d}.png'),0) for body in range(len(cfg['objects']))]
        assert all(mask is not None for mask in full_masks),'Final source mask missing'
        masks=[cv2.resize(mask,(512,384),interpolation=cv2.INTER_NEAREST)>0 for mask in full_masks]
        depths=[renderers[name].depth(pose) for name,pose in zip(cfg['objects'],poses)]
        occlusion=hand_mask(hands,frame)
        visible_hand=cv2.resize(occlusion,(512,384),interpolation=cv2.INTER_NEAREST)==0
        image=cv2.imread(str(rec/f"frames/{frame:06d}.{cfg.get('frame_cache_format','png')}"))
        assert image is not None,'Final source frame missing'
        for body,name in enumerate(cfg['objects']):
            row=lookup[frame,name];pose=poses[body];renderer=renderers[name]
            visible_objects=visibility(masks,depths,body);valid=visible_hand&visible_objects
            measured=measure(renderer.render(pose),masks[body],valid)
            template=templates.get(name)
            other_occlusion=cv2.resize((~visible_objects).astype(np.uint8),image.shape[1::-1],interpolation=cv2.INTER_NEAREST)>0
            observations=(measure_markers(image,template,pose,intrinsics,full_masks[body],
                          (occlusion>0)|other_occlusion,surface_depth=depths[body]) if template else [])
            good=[item for item in observations if item['observed'] and item['measurement']['cost']<4.5]
            if good:
                xyz=[template['markers'][item['marker_id']]['center_m'] for item in good]
                uv=[item['measurement']['xy'] for item in good]
                measured['sphere_error_px']=float(np.median(np.linalg.norm(marker_residual(pose,xyz,uv,intrinsics),axis=1)))
                projected=marker_residual(pose,xyz,np.zeros((len(xyz),2)),intrinsics)
                for item,xy in zip(good,projected):item['fitted_xy']=xy.tolist()
            else:measured['sphere_error_px']=None
            passed=qualifies(measured)
            changed+=bool(passed)!=bool(row.get('local_overlap_pass'))
            row.update(measured,local_overlap_pass=passed,sphere_observations=good,
                       measured_sphere_count=len(good),final_scene_occlusion_rechecked=True,
                       visibility='visible' if measured['visible_pixels']>=60 else 'unverified_occluded',
                       translation_step_m=float(np.linalg.norm(motion['object_to_world'][frame,body,:3,3]-motion['object_to_world'][frame-1,body,:3,3])) if frame else None,
                       rotation_step_rad=float(Rotation.from_matrix(motion['object_to_world'][frame-1,body,:3,:3].T@motion['object_to_world'][frame,body,:3,:3]).magnitude()) if frame else None)
        if progress and (frame%20==0 or frame==cfg['frames']-1):progress('Rechecking source overlays with final scene occlusion',frame+1)
    return dict(method='Recompute silhouettes, cross-object occlusion, hand visibility and locating-sphere reprojection after all object and static-segment pose updates',
                all_object_frames_rechecked=len(records)==cfg['frames']*len(cfg['objects']),
                object_frames_checked=len(records),pass_flags_changed=changed,
                poses_modified=False,source_inputs_modified=False,ground_truth_used=False)
