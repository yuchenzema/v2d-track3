"""Mesh-derived locating spheres and measured, object-associated image centres.

Templates come from textured public scans, in the scan's own metric frame.
Candidates remain labelled as derived features rather than official annotations.
"""
import hashlib
import math
from pathlib import Path
import cv2
import numpy as np
from scipy.optimize import linear_sum_assignment,least_squares
from scipy.spatial import cKDTree
from scipy.sparse import coo_matrix
from scipy.sparse.csgraph import connected_components
import trimesh


def curvature_spheres(vertices,faces):
    """Find sphere centres by normal-line convergence, independent of texture."""
    mesh=trimesh.Trimesh(vertices=vertices,faces=faces,process=True)
    points=np.asarray(mesh.vertices);normals=np.asarray(mesh.vertex_normals)
    if len(points)<80:return []
    tree=cKDTree(points);selected=np.arange(0,len(points),2)
    distances,neighbors=tree.query(points[selected],k=min(40,len(points)))
    p=points[neighbors];n=normals[neighbors];pm=p.mean(1);nm=n.mean(1)
    dp=p-pm[:,None];dn=n-nm[:,None];den=np.sum(dn*dn,axis=(1,2))
    radii=np.sum(dp*dn,axis=(1,2))/np.maximum(den,1e-12)
    centers=pm-radii[:,None]*nm
    residual=np.sqrt(np.mean(np.sum((p-centers[:,None]-radii[:,None,None]*n)**2,axis=2),axis=1))
    valid=(radii>.0025)&(radii<.012)&(residual<.0007)&(np.linalg.norm(nm,axis=1)<.98)&(distances[:,-1]<.01)
    centers=centers[valid];radii=radii[valid]
    if len(centers)<8:return []
    votes=cKDTree(centers);counts=votes.query_ball_point(centers,.0015,return_length=True);result=[]
    for index in np.argsort(-counts):
        if counts[index]<8:break
        if any(np.linalg.norm(centers[index]-np.array(row['center_m']))<.007 for row in result):continue
        group=votes.query_ball_point(centers[index],.0015);center=np.median(centers[group],axis=0);radius=float(np.median(radii[group]))
        indexes=np.array(tree.query_ball_point(center,radius+.0015))
        delta=points[indexes]-center;length=np.linalg.norm(delta,axis=1)
        inliers=(np.abs(length-radius)<.001)&(np.sum(normals[indexes]*delta,axis=1)/np.maximum(length,1e-9)>.65)
        cloud=points[indexes[inliers]]
        if len(cloud)<40:continue
        covariance=np.linalg.eigvalsh(np.cov(cloud.T))
        if covariance[0]/max(covariance[-1],1e-12)<.1:continue
        fit=least_squares(lambda x:np.linalg.norm(cloud-x[:3],axis=1)-x[3],np.r_[center,radius],loss='soft_l1',f_scale=.0003,max_nfev=100)
        radius=float(fit.x[3]);error=np.abs(np.linalg.norm(cloud-fit.x[:3],axis=1)-radius)
        if not .0025<radius<.012 or np.median(error)>.0006:continue
        result.append({'center_m':fit.x[:3].tolist(),'radius_m':radius,'surface_points':len(cloud),
                       'median_surface_residual_m':float(np.median(error)),'p90_surface_residual_m':float(np.quantile(error,.9)),
                       'sphere_center_votes':len(group),'surface_covariance_ratio':float(covariance[0]/covariance[-1]),
                       'status':'mesh_derived_spherical_candidate','derivation':'surface curvature and robust sphere fit'})
    return result


def extract_spheres(path):
    path=Path(path);scene=trimesh.load(path,force='scene',process=False);mesh=scene.to_geometry()
    result={'mesh':str(path),'mesh_sha256':hashlib.sha256(path.read_bytes()).hexdigest(),
            'coordinate_frame':'original scanned mesh, metres','official_marker_annotations':False,
            'method':'Surface-normal convergence and robust 3D sphere fitting; compact grey components as fallback', 'markers':[]}
    curved=curvature_spheres(mesh.vertices,mesh.faces)
    if curved:
        curved.sort(key=lambda row:tuple(row['center_m']))
        for index,row in enumerate(curved):row['marker_id']=index
        result['markers']=curved
        return result
    if mesh.visual.kind!='texture':return result
    colors=mesh.visual.to_color().vertex_colors[:,:3]/255.
    saturation=(colors.max(1)-colors.min(1))/np.maximum(colors.max(1),1e-9)
    points=np.asarray(mesh.vertices)[(saturation<.3)&(colors.mean(1)>.1)&(colors.mean(1)<.75)]
    if not len(points):return result
    _,indexes=np.unique(np.round(points/.0005).astype(int),axis=0,return_index=True);points=points[indexes]
    pairs=cKDTree(points).query_pairs(.002,output_type='ndarray')
    if not len(pairs):return result
    count,labels=connected_components(coo_matrix((np.ones(len(pairs)),(pairs[:,0],pairs[:,1])),shape=(len(points),len(points))),directed=False)
    candidates=[]
    for index in range(count):
        cloud=points[labels==index];span=np.ptp(cloud,axis=0)
        if len(cloud)<35 or span.max()>.035 or span.min()<.0025:continue
        x=np.linalg.lstsq(np.c_[2*cloud,np.ones(len(cloud))],np.sum(cloud*cloud,axis=1),rcond=None)[0]
        radius=math.sqrt(max(0,float(np.sum(x[:3]**2)+x[3])))
        if not .0025<radius<.012:continue
        fit=least_squares(lambda y:np.linalg.norm(cloud-y[:3],axis=1)-y[3],np.r_[x[:3],radius],
                          bounds=(np.r_[cloud.min(0)-.015,.002],np.r_[cloud.max(0)+.015,.015]),
                          loss='soft_l1',f_scale=.0003,max_nfev=100)
        center=fit.x[:3];radius=float(fit.x[3]);error=np.abs(np.linalg.norm(cloud-center,axis=1)-radius)
        if not .0025<radius<.012 or np.median(error)>.0008 or np.quantile(error,.9)>.0018:continue
        candidates.append({'center_m':center.tolist(),'radius_m':radius,'surface_points':len(cloud),
                           'median_surface_residual_m':float(np.median(error)),
                           'p90_surface_residual_m':float(np.quantile(error,.9)),
                           'status':'mesh_derived_spherical_candidate'})
    candidates.sort(key=lambda row:tuple(row['center_m']))
    for index,row in enumerate(candidates):row['marker_id']=index
    result['markers']=candidates
    return result


def project_centers(template,camera_from_object,intrinsics):
    if not template['markers']:return np.empty((0,2)),np.empty(0),np.empty(0)
    points=np.array([m['center_m'] for m in template['markers']]);p=points@camera_from_object[:3,:3].T+camera_from_object[:3,3]
    uv=p[:,:2]/np.maximum(p[:,2:],1e-6)*[intrinsics['fx'],intrinsics['fy']]+[intrinsics['cx'],intrinsics['cy']]
    radii=np.array([m['radius_m'] for m in template['markers']])*math.sqrt(intrinsics['fx']*intrinsics['fy'])/np.maximum(p[:,2],1e-6)
    return uv,radii,p[:,2]


def edge_circle_candidates(image,predicted,radius,search_radius=24):
    """Measure grey cap arcs when the cap merges with grey surroundings.

    Require two Hough thresholds, grey interior and radial edge support over
    most of the circumference. Review coordinates alone never create a cap.
    """
    if not np.isfinite(predicted).all() or not 1<=radius<=18:return []
    height,width=image.shape[:2];x,y=predicted
    x0=max(0,int(x-search_radius));y0=max(0,int(y-search_radius))
    x1=min(width,int(x+search_radius+1));y1=min(height,int(y+search_radius+1))
    roi=image[y0:y1,x0:x1]
    if min(roi.shape[:2],default=0)<8:return []
    grey=cv2.cvtColor(roi,cv2.COLOR_BGR2GRAY)
    blurred=cv2.GaussianBlur(grey,(3,3),.6)
    detections=[]
    for threshold in [10,12]:
        circles=cv2.HoughCircles(blurred,cv2.HOUGH_GRADIENT,1,4,param1=80,param2=threshold,
            minRadius=max(2,int(.4*radius)),maxRadius=max(3,int(2*radius)))
        detections.append([] if circles is None else circles[0])
    hsv=cv2.cvtColor(roi,cv2.COLOR_BGR2HSV)
    gx=cv2.Sobel(grey,cv2.CV_32F,1,0,ksize=3);gy=cv2.Sobel(grey,cv2.CV_32F,0,1,ksize=3)
    angles=np.linspace(0,2*np.pi,36,endpoint=False);directions=np.c_[np.cos(angles),np.sin(angles)]
    yy,xx=np.indices(grey.shape);result=[]
    for a in detections[0]:
        if not any(np.linalg.norm(a[:2]-b[:2])<1.5 and abs(float(a[2]-b[2]))<1.5 for b in detections[1]):continue
        center=a[:2].astype(float);r=float(a[2]);inside=(xx-center[0])**2+(yy-center[1])**2<(.65*r)**2
        if inside.sum()<4:continue
        saturation,value=np.median(hsv[inside,1:3],axis=0)
        if saturation>=85 or not 32<value<235:continue
        radial=center+directions[:,None,:]*r*np.array([.7,.85,1.,1.15,1.3])[None,:,None]
        pixels=np.rint(radial).astype(int);px,py=pixels[:,:,0],pixels[:,:,1]
        in_bounds=(px>=0)&(px<grey.shape[1])&(py>=0)&(py<grey.shape[0])
        px=px.clip(0,grey.shape[1]-1);py=py.clip(0,grey.shape[0]-1)
        gradients=np.stack([gx[py,px],gy[py,px]],axis=-1);length=np.linalg.norm(gradients,axis=-1)
        agreement=np.abs(np.sum(gradients*directions[:,None,:],axis=-1))/np.maximum(length,1e-9)
        coverage=float(np.mean(np.any(in_bounds&(length>30)&(agreement>.7),axis=1)))
        if coverage<.7:continue
        full=center+[x0,y0]
        cost=float(np.sum((full-predicted)**2)/100.+2*math.log(r/radius)**2+1-coverage)
        result.append(dict(xy=full.tolist(),radius_px=r,circularity=coverage,cost=cost,
            method='measured_stable_radial_grey_cap',edge_arc_coverage=coverage,hough_thresholds=[10,12]))
    return result


def circle_candidates(image,predicted,radius,search_radius=24):
    if not np.isfinite(predicted).all() or not 1<=radius<=18:return []
    height,width=image.shape[:2];x,y=predicted
    x0=max(0,int(x-search_radius));y0=max(0,int(y-search_radius));x1=min(width,int(x+search_radius+1));y1=min(height,int(y+search_radius+1))
    if x1-x0<8 or y1-y0<8:return []
    roi=image[y0:y1,x0:x1];hsv=cv2.cvtColor(roi,cv2.COLOR_BGR2HSV)
    def components(mask):
        contours,_=cv2.findContours(mask,cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_SIMPLE);result=[]
        for contour in contours:
            area=cv2.contourArea(contour);perimeter=cv2.arcLength(contour,True)
            if area<3 or perimeter<=0:continue
            circularity=4*math.pi*area/(perimeter*perimeter)
            if circularity<.48:continue
            moments=cv2.moments(contour)
            if moments['m00']<=0:continue
            center=np.array([moments['m10']/moments['m00']+x0,moments['m01']/moments['m00']+y0]);r=math.sqrt(area/math.pi)
            if not .4*radius<=r<=2.*radius:continue
            if np.linalg.norm(center-predicted)>search_radius:continue
            cost=float(np.sum((center-predicted)**2)/100.+2.*math.log(r/radius)**2+(1-circularity))
            result.append({'xy':center.tolist(),'radius_px':r,'circularity':circularity,'cost':cost,'method':'measured_low_saturation_circular_component'})
        return result
    mask=((hsv[:,:,1]<85)&(hsv[:,:,2]>32)&(hsv[:,:,2]<235)).astype(np.uint8)
    original=components(mask)
    # An unrelated circular feature can lie inside the search region but
    # already fail the fitter's unchanged association-cost limit. It must not
    # suppress the multithreshold search for the actual grey cap.
    if original and min(item['cost'] for item in original)<4.5:return original
    # Grey caps can merge with white plastic or narrow dark marker stems at a
    # single threshold. Require measured cap agreement at multiple intensities.
    groups=[]
    # Bright grey caps can be separated from wood/white plastic only between
    # 192 and 224. Sample that interval as well, while still requiring two
    # independently thresholded components with the same measured centre.
    for upper in range(96,225,16):
        binary=((hsv[:,:,1]<85)&(hsv[:,:,2]>32)&(hsv[:,:,2]<upper)).astype(np.uint8)
        for size in (0,3,5):
            opened=(cv2.morphologyEx(binary,cv2.MORPH_OPEN,cv2.getStructuringElement(cv2.MORPH_ELLIPSE,(size,size))) if size else binary)
            for item in components(opened):
                group=next((g for g in groups if np.linalg.norm(np.array(item['xy'])-np.median([a['xy'] for a in g['items']],axis=0))<3.),None)
                if group is None:group=dict(items=[],levels=set(),scales=set());groups.append(group)
                group['items'].append(item);group['levels'].add(upper);group['scales'].add(size)
    result=[]
    for group in groups:
        if len(group['levels'])<2:continue
        center=np.median([a['xy'] for a in group['items']],axis=0);r=float(np.median([a['radius_px'] for a in group['items']]));circularity=float(np.median([a['circularity'] for a in group['items']]))
        cost=float(np.sum((center-predicted)**2)/100.+2.*math.log(r/radius)**2+(1-circularity))
        result.append(dict(xy=center.tolist(),radius_px=r,circularity=circularity,cost=cost,
            method='measured_multithreshold_grey_cap',intensity_levels=sorted(group['levels']),opening_scales=sorted(group['scales'])))
    measured=original+result
    if measured and min(item['cost'] for item in measured)<4.5:return measured
    return measured+edge_circle_candidates(image,predicted,radius,search_radius)


def measure_markers(image,template,camera_from_object,intrinsics,object_mask=None,hand_occlusion=None,surface_depth=None):
    predictions,radii,depth=project_centers(template,camera_from_object,intrinsics)
    height,width=image.shape[:2];all_candidates=[];candidate_by_marker=[]
    allowed=cv2.dilate(object_mask.astype(np.uint8),np.ones((25,25),np.uint8)) if object_mask is not None else None
    self_occluded=set();hand_occluded=set()
    for index,prediction in enumerate(predictions):
        choices=[]
        if hand_occlusion is not None and np.isfinite(prediction).all():
            x,y=np.rint(prediction).astype(int)
            if 0<=x<width and 0<=y<height and hand_occlusion[y,x]:
                hand_occluded.add(index);candidate_by_marker.append(choices);continue
        if surface_depth is not None and np.isfinite(prediction).all():
            sh,sw=surface_depth.shape;x,y=np.rint(prediction*[sw/width,sh/height]).astype(int)
            if 0<=x<sw and 0<=y<sh:
                front=float(surface_depth[y,x]);radius=float(template['markers'][index]['radius_m'])
                if front>.05 and depth[index]-radius>front+.004:
                    self_occluded.add(index);candidate_by_marker.append(choices);continue
        if depth[index]>.05 and -24<prediction[0]<width+24 and -24<prediction[1]<height+24:
            for candidate in circle_candidates(image,prediction,radii[index]):
                x,y=np.rint(candidate['xy']).astype(int)
                if not 0<=x<width or not 0<=y<height:continue
                if allowed is not None and not allowed[y,x]:continue
                if hand_occlusion is not None and hand_occlusion[y,x]:continue
                existing=next((j for j,c in enumerate(all_candidates) if np.linalg.norm(np.array(c['xy'])-candidate['xy'])<1.5),None)
                if existing is None:existing=len(all_candidates);all_candidates.append(candidate)
                choices.append((existing,candidate['cost']))
        candidate_by_marker.append(choices)
    if not len(predictions):return []
    costs=np.full((len(predictions),len(all_candidates)+len(predictions)),6.5)
    costs[:,:len(all_candidates)]=1e6
    for index,choices in enumerate(candidate_by_marker):
        for j,cost in choices:costs[index,j]=min(costs[index,j],cost)
    rows,columns=linear_sum_assignment(costs);observed={}
    for i,j in zip(rows,columns):
        if j<len(all_candidates) and costs[i,j]<6.5:
            observed[i]={**all_candidates[j],'cost':float(costs[i,j])}
    return [{'marker_id':i,'observed':i in observed,'visibility':'hand_occluded' if i in hand_occluded else 'self_occluded' if i in self_occluded else 'candidate',
             'predicted_xy':p.tolist(),'predicted_radius_px':float(radii[i]),
             **({'measurement':observed[i]} if i in observed else {})} for i,p in enumerate(predictions)]


def marker_residual(camera_from_object,centers,observations,intrinsics):
    centers=np.asarray(centers);points=centers@camera_from_object[:3,:3].T+camera_from_object[:3,3]
    projected=points[:,:2]/np.maximum(points[:,2:],.01)*[intrinsics['fx'],intrinsics['fy']]+[intrinsics['cx'],intrinsics['cy']]
    return projected-np.asarray(observations)
