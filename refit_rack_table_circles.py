"""Refit occluded tabletop arcs without locking to an erroneous first-frame circle."""
import json,cv2,numpy as np
from pathlib import Path
from scipy.optimize import least_squares
R=Path('/home/yu_eli/v2d-track3/runs/episode_000014_newton_batch_20261015');rec=R/'reconstruction';cfg=json.loads((R/'episode-config.json').read_text());intr=json.loads((rec/'camera_intrinsics.json').read_text());planes=np.load(rec/'table-camera-calibration.npz')['table_planes_camera'];independent=json.loads((R/'evaluation/rack-based-table-diagnostic.json').read_text());radius_prior=float(np.median([x['radius_m'] for x in independent['frames'] if x['inlier_fraction']>.25 and .2<x['radius_m']<.3]));rng=np.random.default_rng(42);cap=cv2.VideoCapture(str(rec/'calibrated.mp4'));centers=[];radii=[];scores=[];panels=[]
for i,plane in enumerate(planes):
 ok,im=cap.read();assert ok;hsv=cv2.cvtColor(im,cv2.COLOR_BGR2HSV);wood=((hsv[:,:,0]>3)&(hsv[:,:,0]<30)&(hsv[:,:,1]>90)&(hsv[:,:,2]>50)).astype('uint8')*255
 for j in range(len(cfg['objects'])):wood[cv2.dilate(cv2.imread(str(rec/f'masks/{j+1}/{i:06d}.png'),0),np.ones((7,7),np.uint8))>0]=0
 contours,_=cv2.findContours(wood,cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_NONE);candidates=[c for c in contours if cv2.contourArea(c)>100 and c[:,:,1].mean()>350];contour=max(candidates,key=cv2.contourArea)[:,0,:][::2];contour=contour[contour[:,1]>300];n,d=plane[:3],plane[3];rays=np.c_[(contour[:,0]-intr['cx'])/intr['fx'],(contour[:,1]-intr['cy'])/intr['fy'],np.ones(len(contour))];distance=d/(rays@n);points=rays*distance[:,None];x=np.array([1.,0.,0.]);x-=n*(x@n);x/=np.linalg.norm(x);basis=np.c_[x,np.cross(n,x)];base=n*d;p=(points-base)@basis;p=p[(distance>.2)&(distance<1.3)];best=None
 for _ in range(1800):
  a,b,c=p[rng.choice(len(p),3,replace=False)];A=2*np.array([b-a,c-a]);rhs=np.array([b@b-a@a,c@c-a@a])
  try:center=np.linalg.solve(A,rhs)
  except np.linalg.LinAlgError:continue
  radius=np.linalg.norm(a-center)
  if abs(radius-radius_prior)>.025:continue
  error=abs(np.linalg.norm(p-center,axis=1)-radius);hit=error<.007
  if best is None or hit.sum()>best[0]:best=(hit.sum(),center,radius,hit)
 assert best is not None,i
 _,center,radius,hit=best;fit=least_squares(lambda z:np.linalg.norm(p[hit]-z[:2],axis=1)-z[2],np.r_[center,radius],bounds=([-np.inf,-np.inf,radius_prior-.025],[np.inf,np.inf,radius_prior+.025]),loss='soft_l1',f_scale=.003);center_camera=base+basis@fit.x[:2];centers.append(center_camera);radii.append(float(fit.x[2]));scores.append(float(hit.mean()))
 if i in set(np.linspace(0,len(planes)-1,8).round().astype(int)):
  angles=np.linspace(0,2*np.pi,200);v=center_camera+fit.x[2]*(np.cos(angles)[:,None]*basis[:,0]+np.sin(angles)[:,None]*basis[:,1]);uv=np.rint(v[:,:2]/v[:,2:]*[intr['fx'],intr['fy']]+[intr['cx'],intr['cy']]).astype(int);cv2.polylines(im,[uv],True,(0,255,0),3);cv2.putText(im,f'{i} r={fit.x[2]:.3f}',(15,40),0,1,(0,255,0),2);panels.append(cv2.resize(im,(512,384)))
 if i%30==0:print(i,radii[-1],scores[-1],flush=True)
cap.release();np.savez_compressed(rec/'refitted-table-circle-measurements.npz',table_centers_camera=centers,measured_radii=radii,circle_inlier_fraction=scores);cv2.imwrite(str(rec/'refitted-table-circle-review.jpg'),np.concatenate([np.concatenate(panels[j:j+4],axis=1) for j in [0,4]]));(rec/'refitted-table-circle-report.json').write_text(json.dumps({'method':__doc__,'independent_radius_prior_m':radius_prior,'prior_source':'Released rack geometry and visible wood edges in eight original frames; no evaluation pose labels.','radius_median_m':float(np.median(radii)),'radius_p05_p95_m':np.quantile(radii,[.05,.95]).tolist(),'median_circle_inlier_fraction':float(np.median(scores))},indent=2))
