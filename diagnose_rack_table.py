"""Independent tabletop metric check using released rack dimensions and original RGB wood edges."""
from pathlib import Path
import json,cv2,numpy as np,trimesh
from scipy.spatial.transform import Rotation
from scipy.optimize import least_squares
BASE=Path('/home/yu_eli/v2d-track3')
for ep in [14,15,16,28]:
 r=BASE/'runs'/f'episode_{ep:06d}_newton_batch_20261015';rec=r/'reconstruction';c=json.loads((r/'episode-config.json').read_text());b=c['objects'].index('plastic_dish_rack');m=np.load(rec/'scene_motion_physical.npz');intr=json.loads((rec/'camera_intrinsics.json').read_text());mesh=trimesh.load(rec/'mesh_priors/plastic_dish_rack.ply',process=False);cap=cv2.VideoCapture(str(rec/'calibrated.mp4'));report=[]
 for i in np.linspace(0,c['frames']-1,8).round().astype(int):
  cap.set(cv2.CAP_PROP_POS_FRAMES,int(i));ok,im=cap.read();assert ok;M=np.linalg.inv(m['camera_to_world'][i])@m['object_to_world'][i,b];normal=M[:3,0];base=M[:3,3]+normal*np.quantile(mesh.vertices[:,0],.01);axis=M[:3,1:3];hsv=cv2.cvtColor(im,cv2.COLOR_BGR2HSV);wood=((hsv[:,:,0]>3)&(hsv[:,:,0]<30)&(hsv[:,:,1]>90)&(hsv[:,:,2]>50)).astype('uint8')*255
  for j in range(len(c['objects'])):
   mask=cv2.imread(str(rec/f'masks/{j+1}/{i:06d}.png'),0);wood[cv2.dilate(mask,np.ones((7,7),np.uint8))>0]=0
  contours,_=cv2.findContours(wood,cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_NONE);contour=max(contours,key=cv2.contourArea);uv=contour[:,0,:];uv=uv[uv[:,1]>300];rays=np.c_[(uv[:,0]-intr['cx'])/intr['fx'],(uv[:,1]-intr['cy'])/intr['fy'],np.ones(len(uv))];distance=(normal@base)/(rays@normal);p=rays*distance[:,None];xy=(p-base)@axis;valid=(distance>.2)&(distance<1.4)&(np.linalg.norm(xy,axis=1)<1.);xy=xy[valid];rng=np.random.default_rng(42);best=None
  for _ in range(1500):
   a0,a1,a2=xy[rng.choice(len(xy),3,replace=False)];A=2*np.array([a1-a0,a2-a0]);rhs=np.array([a1@a1-a0@a0,a2@a2-a0@a0])
   try:center=np.linalg.solve(A,rhs)
   except np.linalg.LinAlgError:continue
   radius=np.linalg.norm(a0-center)
   if not .15<radius<.45:continue
   error=abs(np.linalg.norm(xy-center,axis=1)-radius);hits=error<.007
   if best is None or hits.sum()>best[0]:best=(hits.sum(),center,radius,hits)
  if best is None:continue
  _,center,radius,hits=best;fit=least_squares(lambda x:np.linalg.norm(xy[hits]-x[:2],axis=1)-x[2],np.r_[center,radius]);report.append({'frame':int(i),'radius_m':float(fit.x[2]),'inlier_fraction':float(hits.mean()),'rack_plane_to_stereo_table_normal_deg':float(np.degrees(np.arccos(np.clip(abs(normal@np.load(rec/'table-camera-calibration.npz')['table_planes_camera'][i,:3]),0,1))))})
 cap.release();(r/'evaluation/rack-based-table-diagnostic.json').write_text(json.dumps({'method':__doc__,'frames':report},indent=2));print(ep,report,flush=True)
