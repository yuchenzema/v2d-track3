"""Recover missing spoon sphere candidates from actual metric stereo samples.

No RGB-depth prior is substituted for missing stereo. Candidates need repeated
cross-frame metric agreement in the existing reconstructed spoon frame.
"""
import cv2
import numpy as np
from scipy.spatial import cKDTree
from improvement_common import save,utc


def recover(source,cfg,motion,intr,root):
    path=root/'research/video-derived-spoon-spheres.json'
    if path.exists():
        import json
        return json.loads(path.read_text())
    rec=source/'reconstruction';body=cfg['objects'].index('wooden_spoon');points=[];observations=[]
    for frame in range(0,cfg['frames'],3):
        image=cv2.imread(str(rec/f"frames/{frame:06d}.{cfg.get('frame_cache_format','png')}"))
        raw=cv2.imread(str(rec/f'depth_stereo/{frame:06d}.png'),-1)
        mask=cv2.imread(str(rec/f'masks/{body+1}/{frame:06d}.png'),0)
        if image is None or raw is None or mask is None:continue
        hsv=cv2.cvtColor(image,cv2.COLOR_BGR2HSV)
        grey=((hsv[:,:,1]<85)&(hsv[:,:,2]>32)&(hsv[:,:,2]<235)&(mask>0)).astype(np.uint8)
        contours,_=cv2.findContours(grey,cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_SIMPLE)
        object_from_camera=np.linalg.inv(motion['object_to_world'][frame,body])@motion['camera_to_world'][frame]
        for contour in contours:
            area=cv2.contourArea(contour);perimeter=cv2.arcLength(contour,True)
            if not 6<area<160 or perimeter<=0 or 4*np.pi*area/perimeter**2<.60:continue
            m=cv2.moments(contour)
            if m['m00']<=0:continue
            uv=np.array([m['m10']/m['m00'],m['m01']/m['m00']]);u,v=np.rint(uv).astype(int)
            patch=raw[max(0,v-2):v+3,max(0,u-2):u+3];values=patch[patch>0]
            if len(values)<6:continue
            depths=65535/values.astype(float)-1;z=float(np.median(depths))
            if not .15<z<1.5 or np.quantile(depths,.9)-np.quantile(depths,.1)>.012:continue
            radius=float(np.sqrt(area/np.pi)*z/np.sqrt(intr['fx']*intr['fy']))
            if not .0025<radius<.012:continue
            ray=np.r_[(uv-[intr['cx'],intr['cy']])/[intr['fx'],intr['fy']],1.]
            center=ray*(z+radius);local=object_from_camera[:3,:3]@center+object_from_camera[:3,3]
            points.append(local);observations.append(dict(frame=frame,xy=uv.tolist(),radius_m=radius,
                   stereo_surface_depth_m=z,stereo_surface_spread_m=float(np.std(depths)),center_m=local.tolist()))
    candidates=[]
    if points:
        points=np.array(points);tree=cKDTree(points);counts=tree.query_ball_point(points,.007,return_length=True)
        for index in np.argsort(-counts):
            neighbors=tree.query_ball_point(points[index],.007)
            frames={observations[n]['frame'] for n in neighbors}
            if len(frames)<5:continue
            center=np.median(points[neighbors],axis=0)
            if any(np.linalg.norm(center-np.array(c['center_m']))<.012 for c in candidates):continue
            spread=float(np.median(np.linalg.norm(points[neighbors]-center,axis=1)))
            if spread>.0035:continue
            candidates.append(dict(marker_id=len(candidates),center_m=center.tolist(),
                 radius_m=float(np.median([observations[n]['radius_m'] for n in neighbors])),
                 observed_source_frames=sorted(frames),metric_center_spread_m=spread,
                 status='video_stereo_derived_sphere_candidate_requires_projection_validation'))
    result=dict(updated_utc=utc(),markers=candidates,observations=observations,
                official_marker_annotations=False,coordinate_frame='episode reconstructed spoon mesh, metres',
                method=__doc__,stereo_only=True,
                limitation='Initial pose uncertainty and sphere surface-to-centre offset affect reconstruction. Missing candidates are not filled.')
    save(path,result);return result
