"""Source-only pose hypotheses from independently detected stereo spheres."""
from itertools import combinations,permutations
import cv2
import numpy as np
from scipy.spatial.transform import Rotation


def detect(image,object_mask,hand_occlusion,depth,intrinsics,template):
    """Measure grey sphere caps before consulting any candidate object pose."""
    if not template or len(template.get('markers',[]))<2:return []
    mask=object_mask>0
    allowed=cv2.dilate(mask.astype(np.uint8),np.ones((3,3),np.uint8)).astype(bool)&~(hand_occlusion>0)
    metric=depth[mask&np.isfinite(depth)&(depth>.05)&(depth<3.)]
    if len(metric)<20:return []
    z=float(np.median(metric));focal=np.sqrt(intrinsics['fx']*intrinsics['fy'])
    radii=np.array([m['radius_m'] for m in template['markers']])*focal/z
    minimum=max(1.2,.5*float(radii.min()));maximum=min(18.,1.7*float(radii.max()))
    hsv=cv2.cvtColor(image,cv2.COLOR_BGR2HSV)
    binary=((hsv[:,:,1]<85)&(hsv[:,:,2]>64)&(hsv[:,:,2]<235)&allowed).astype(np.uint8)
    groups=[]
    # Opening removes narrow stems and bristles attached to grey sphere caps.
    # A cap must survive both scales; one-scale bristle blobs are not markers.
    for size in (3,5):
        opened=cv2.morphologyEx(binary,cv2.MORPH_OPEN,cv2.getStructuringElement(cv2.MORPH_ELLIPSE,(size,size)))
        contours,_=cv2.findContours(opened,cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_SIMPLE)
        for contour in contours:
            area=cv2.contourArea(contour);perimeter=cv2.arcLength(contour,True)
            if area<6 or perimeter<=0:continue
            radius=np.sqrt(area/np.pi);circularity=4*np.pi*area/perimeter**2
            if not minimum<=radius<=maximum or circularity<.5:continue
            moment=cv2.moments(contour)
            if moment['m00']<=0:continue
            xy=np.array([moment['m10']/moment['m00'],moment['m01']/moment['m00']])
            group=next((g for g in groups if np.linalg.norm(xy-np.mean(g['xy'],axis=0))<2.5),None)
            if group is None:
                group=dict(xy=[],radius=[],circularity=[],scales=set());groups.append(group)
            group['xy'].append(xy);group['radius'].append(radius);group['circularity'].append(circularity);group['scales'].add(size)
    observations=[]
    for group in groups:
        if len(group['scales'])<2:continue
        xy=np.mean(group['xy'],axis=0);radius=float(np.median(group['radius']));x,y=np.rint(xy).astype(int)
        if not (0<=y<mask.shape[0] and 0<=x<mask.shape[1]) or not allowed[y,x]:continue
        extent=max(1,int(radius*.6));y0=max(0,y-extent);x0=max(0,x-extent)
        patch=depth[y0:y+extent+1,x0:x+extent+1];yy,xx=np.indices(patch.shape)
        circle=(xx+x0-x)**2+(yy+y0-y)**2<=extent**2
        values=patch[circle&np.isfinite(patch)&(patch>.05)&(patch<3.)]
        if len(values)<5:continue
        value=float(np.median(values));mad=float(np.median(np.abs(values-value)))
        if mad>.012:continue
        ray=np.r_[(xy-[intrinsics['cx'],intrinsics['cy']])/[intrinsics['fx'],intrinsics['fy']],1.]
        observations.append(dict(xy=xy.tolist(),radius_px=radius,surface_xyz_m=(ray*value).tolist(),
            unit_ray=(ray/np.linalg.norm(ray)).tolist(),depth_m=value,depth_mad_m=mad,
            circularity=float(np.mean(group['circularity'])),opening_scales=sorted(group['scales']),
            method='pose_independent_grey_cap_and_stereo_depth'))
    return sorted(observations,key=lambda o:(o['depth_mad_m'],-o['circularity']))[:6]


def pose_candidates(template,observations,roll_samples=24,max_pair_error_m=.012,maximum_pairs=12):
    """Two sphere centres determine a metric axis; sweep its remaining roll."""
    markers=template.get('markers',[]) if template else []
    if len(markers)<2 or len(observations)<2:return []
    pairs=[]
    for observed in combinations(range(len(observations)),2):
        for matched in permutations(range(len(markers)),2):
            points=np.array([markers[m]['center_m'] for m in matched],float)
            centres=np.array([np.asarray(observations[o]['surface_xyz_m'])+
                markers[m]['radius_m']*np.asarray(observations[o]['unit_ray']) for o,m in zip(observed,matched)])
            length=np.linalg.norm(points[1]-points[0]);measured=np.linalg.norm(centres[1]-centres[0])
            error=abs(length-measured)
            if min(length,measured)<.01 or error>max_pair_error_m:continue
            pairs.append((float(error),observed,matched,points,centres))
    candidates=[]
    for error,observed,matched,points,centres in sorted(pairs,key=lambda p:p[0])[:maximum_pairs]:
        axis=centres[1]-centres[0];axis/=np.linalg.norm(axis)
        initial=Rotation.align_vectors(axis[None],((points[1]-points[0])/np.linalg.norm(points[1]-points[0]))[None])[0].as_matrix()
        for angle in np.arange(roll_samples)*2*np.pi/roll_samples:
            rotation=Rotation.from_rotvec(axis*angle).as_matrix()@initial
            pose=np.eye(4);pose[:3,:3]=rotation;pose[:3,3]=centres.mean(0)-rotation@points.mean(0)
            candidates.append(dict(pose=pose,marker_ids=list(matched),observation_ids=list(observed),
                                   pair_distance_error_m=error,roll_rad=float(angle)))
    return candidates
