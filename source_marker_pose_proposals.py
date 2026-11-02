"""Source-image reviewed cap locations; no pose labels or frame retiming."""
from itertools import permutations
import json
from pathlib import Path

import cv2
import numpy as np
from scipy.spatial.transform import Rotation

from improvement_common import BATCH,digest
from sphere_marker_reference import circle_candidates,measure_markers,marker_residual
from contour_pose_refinement import propose as contour_pose_proposal


def load_reviews(cfg,rec,directory=None):
    directory=Path(directory) if directory else BATCH/'focused_episodes_20261028/source-marker-overrides'
    path=directory/f"{cfg.get('dataset_split','evaluation')}_{cfg['episode']:06d}.json"
    if not path.is_file():return [],None
    data=json.loads(path.read_text())
    if data.get('episode')!=cfg['episode'] or data.get('dataset_split')!=cfg.get('dataset_split','evaluation'):
        raise ValueError('Source-marker review belongs to another episode or split')
    reviews=data['reviews']
    for row in reviews:
        frame=row['frame'];image=Path(rec)/f"frames/{frame:06d}.{cfg.get('frame_cache_format','png')}"
        if row['object'] not in cfg['objects'] or not 0<=frame<cfg['frames'] or digest(image)!=row['source_frame_sha256']:
            raise ValueError('Source-marker review is not bound to the current original source frame')
        if not 3<=len(row['points'])<=4:raise ValueError('Reviewed PnP needs three or four independent image caps')
    return reviews,dict(path=str(path),sha256=digest(path),ground_truth_used=False)


def checked_observations(image,template,pose,intr,mask,occlusion,renderer,target,valid,measure,qualifies):
    observations=measure_markers(image,template,pose,intr,mask,occlusion,surface_depth=renderer.depth(pose))
    good=[o for o in observations if o['observed'] and o['measurement']['cost']<4.5]
    if len(good)<3:return None
    xyz=[template['markers'][o['marker_id']]['center_m'] for o in good]
    uv=[o['measurement']['xy'] for o in good]
    row=measure(renderer.render(pose),target,valid)
    row['sphere_error_px']=float(np.median(np.linalg.norm(marker_residual(pose,xyz,uv,intr),axis=1)))
    return (observations,row) if qualifies(row) else None


def bootstrap(image,template,renderer,intr,mask,occlusion,target,valid,points,measure,qualifies):
    """Enumerate cap IDs, then remeasure and validate the proposed front surface."""
    markers=template.get('markers',[]);measured=[]
    if not 3<=len(points)<=4 or not len(points)<=len(markers)<=6:return []
    for point in points:
        choices=[o for o in circle_candidates(image,np.array(point['xy'],float),point['radius_px']) if o['cost']<4.5]
        if not choices:return []
        chosen=min(choices,key=lambda o:o['cost'])
        if any(np.linalg.norm(np.array(chosen['xy'])-other['xy'])<3 for other in measured):return []
        measured.append(chosen)
    k=np.array([[intr['fx'],0,intr['cx']],[0,intr['fy'],intr['cy']],[0,0,1.]])
    def solve(xyz,uv):
        try:
            _,rvecs,tvecs,_=cv2.solvePnPGeneric(np.array(xyz,float),np.array(uv,float),k,None,flags=cv2.SOLVEPNP_SQPNP)
        except cv2.error:return []
        poses=[]
        for rv,tv in zip(rvecs,tvecs):
            pose=np.eye(4);pose[:3,:3]=cv2.Rodrigues(rv)[0];pose[:3,3]=tv.ravel()
            if np.isfinite(pose).all() and pose[2,3]>.05:poses.append(pose)
        return poses
    proposals=[]
    for ids in permutations(range(len(markers)),len(measured)):
        poses=solve([markers[i]['center_m'] for i in ids],[o['xy'] for o in measured])
        for start in poses:
            if measure(renderer.render(start),target,valid)['iou']<.8:continue
            # Measured sphere centres are pixel estimates. A small cap-centre
            # error can leave the exact PnP solution outside the body contour
            # gate. Refine against the source silhouette, then remeasure all
            # front-visible caps and apply the original full source gates.
            checked=checked_observations(image,template,start,intr,mask,occlusion,renderer,target,valid,measure,qualifies)
            if not checked:
                refined=contour_pose_proposal(renderer,start,target,valid,intr)
                if refined is not None:
                    refined_check=checked_observations(image,template,refined,intr,mask,occlusion,renderer,target,valid,measure,qualifies)
                    if refined_check:proposals.append(dict(pose=refined.copy(),observations=refined_check[0],metrics=refined_check[1]))
            pose=start
            if checked:proposals.append(dict(pose=pose.copy(),observations=checked[0],metrics=checked[1]))
            for _ in range(3):
                observations=measure_markers(image,template,pose,intr,mask,occlusion,surface_depth=renderer.depth(pose))
                good=[o for o in observations if o['observed'] and o['measurement']['cost']<4.5]
                if len(good)<3:break
                options=solve([markers[o['marker_id']]['center_m'] for o in good],[o['measurement']['xy'] for o in good])
                if not options:break
                pose=max(options,key=lambda p:measure(renderer.render(p),target,valid)['iou'])
                # Reassociation can move a measured cap onto a nearby stem or
                # edge. Keep every source-verified iterate; a later regression
                # must not erase an already valid source-image proposal.
                checked=checked_observations(image,template,pose,intr,mask,occlusion,renderer,target,valid,measure,qualifies)
                if checked:proposals.append(dict(pose=pose.copy(),observations=checked[0],metrics=checked[1]))
    proposals.sort(key=lambda p:-p['metrics']['iou'])
    if not proposals:return []
    # Do not resolve an image-ambiguous orientation by an arbitrary permutation.
    best=proposals[0]
    for other in proposals[1:]:
        angle=Rotation.from_matrix(best['pose'][:3,:3].T@other['pose'][:3,:3]).magnitude()
        if angle>.3 and best['metrics']['iou']-other['metrics']['iou']<.005:return []
    return proposals
