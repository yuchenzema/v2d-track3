"""Carry table position through occlusion using measured metric camera motion."""
import numpy as np
from scipy.optimize import least_squares


def refine_circle(points, center, radius):
    """Keep refinement inside the measured RANSAC neighbourhood and size prior.

    A short nearly straight arc cannot determine an unconstrained circle radius.
    Never allow that numerical ambiguity to create a hundreds-of-metres anchor.
    """
    points=np.asarray(points,float);center=np.asarray(center,float)
    if len(points)<6 or not np.isfinite(points).all():
        raise ValueError('Insufficient finite table-circle edge measurements')
    if not .18<radius<.36:raise ValueError('Table circle seed is outside the size prior')
    result=least_squares(lambda z:np.linalg.norm(points-z[:2],axis=1)-z[2],
        np.r_[center,radius],bounds=(np.r_[center-.04,.18],np.r_[center+.04,.36]),
        loss='soft_l1',f_scale=.003,max_nfev=100)
    if not result.success or not np.isfinite(result.x).all():
        raise ValueError('Table-circle refinement failed')
    return result.x


def table_component(mask):
    """Retain the table's connected colour region, excluding detached skin/background."""
    import cv2
    binary=cv2.morphologyEx(np.asarray(mask,np.uint8),cv2.MORPH_CLOSE,np.ones((9,9),np.uint8))
    count,labels,stats,_=cv2.connectedComponentsWithStats(binary)
    if count<=1:return np.zeros_like(mask,dtype=bool)
    label=1+np.argmax(stats[1:,cv2.CC_STAT_AREA])
    return (labels==label)&np.asarray(mask,bool)


def propagate_center(center,previous_camera,current_camera,normal=None,plane_d=None):
    relative=np.linalg.inv(current_camera)@previous_camera
    result=relative[:3,:3]@np.asarray(center)+relative[:3,3]
    if normal is not None and plane_d is not None:
        normal=np.asarray(normal,float);normal=normal/np.linalg.norm(normal)
        result+=normal*(float(plane_d)-normal@result)
    if not np.isfinite(result).all():raise ValueError('Measured camera motion cannot propagate the table center')
    return result


def adapt(source):
    source=source.replace('ROOT=Path(__file__).resolve().parents[1];REC=',
        "import sys\nsys.path.insert(0,str(Path(__file__).resolve().parents[2].parent))\nfrom robust_table_camera_template import propagate_center,refine_circle,table_component\nROOT=Path(__file__).resolve().parents[1];REC=")
    source=source.replace(' valid=wood&stereo', ' wood=table_component(wood)\n valid=wood&stereo')
    old="q=least_squares(lambda z:np.linalg.norm(p[best_hit]-z[:2],axis=1)-z[2],np.r_[best_center,best_radius],loss='soft_l1',f_scale=.003).x"
    assert old in source
    source=source.replace(old,'q=refine_circle(p[best_hit],best_center,best_radius)')
    source=source.replace('scores=[];debug=[]','scores=[];debug=[];circle_fallback_frames=[]')
    source=source.replace('else:n=prior;d=d0',
        '''else:
  n=prior
  predicted=propagate_center(centers[-1],slam[i-1],slam[i]) if centers else n*d0
  d=float(n@predicted)''')
    old='contour=max(contours,key=cv2.contourArea)[:,0][::2].astype(float)'
    assert old in source
    source=source.replace(old,old+' if contours else np.empty((0,2))')
    source=source.replace('for _ in range(250):','for _ in range(250 if len(p)>=3 else 0):')
    old=" if best_center is None:\n  raise RuntimeError(f'Cannot measure table circle in frame {i}')"
    assert old in source
    source=source.replace(old,''' if best_center is None:
  if not centers:raise RuntimeError(f'Initial table anchor is not measurable in frame {i}')
  center_camera=propagate_center(centers[-1],slam[i-1],slam[i],n if best.sum()>50 else None,d if best.sum()>50 else None)
  planes.append(np.r_[n,d]);centers.append(center_camera);radii.append(radii[-1]);scores.append(0.);circle_fallback_frames.append(i)
  print('Table circle occluded; measured camera-motion fallback',i,flush=True)
  if i in set(np.linspace(0,T-1,8).round().astype(int)):
   cv2.putText(im,'Table center from measured camera motion',(12,35),0,.7,(0,180,255),2);debug.append(cv2.resize(im,(512,384)))
  continue''')
    source=source.replace("'frame_count':T,","'frame_count':T,'circle_fallback_frames':circle_fallback_frames,'all_frame_circles_directly_measured':not circle_fallback_frames,'circle_fallback_method':'Measured metric SLAM relative camera motion, constrained by the current stereo table plane when available; no task object assumed static',")
    source=source.replace("'frame_count':T,", "'circle_refinement_version':2,'circle_radius_bounds_m':[.18,.36],'frame_count':T,")
    source=source.replace("'translation_source':'per-frame static table circle and stereo plane; no object assumed static during lifts'",
        "'translation_source':'Observed table circle and stereo plane; explicitly marked circle-occluded frames use measured metric SLAM motion'")
    return source
