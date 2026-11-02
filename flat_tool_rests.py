"""Fit a stationary flat tool directly to clear source silhouettes on the measured tabletop."""
from pathlib import Path
import importlib.util,json
import cv2,numpy as np
from scipy.ndimage import distance_transform_edt,map_coordinates
from scipy.optimize import least_squares
from scipy.spatial import cKDTree
from scipy.spatial.transform import Rotation

def fit(root,vertices,center,cameras,anchor_indices,object_index,initial_rotations,fixed_rotation=None,support_vertices=None):
 root=Path(root);rec=root/'reconstruction';cfg=json.loads((root/'episode-config.json').read_text());intr=json.loads((rec/'camera_intrinsics.json').read_text());spec=importlib.util.spec_from_file_location('rest_source_geometry',root/'scripts/source_geometry.py');g=importlib.util.module_from_spec(spec);spec.loader.exec_module(g);measure=np.load(rec/'hand_measurements.npz');rotation=Rotation.from_matrix(initial_rotations).mean().as_matrix();normal=rotation[:,0];target=np.array([0.,0.,1. if normal[2]>=0 else -1.]);align,_=Rotation.align_vectors([target],[normal]);rotation=align.as_matrix()@rotation;height=.6505+float((rotation@center)[2])-(vertices@rotation.T)[:,2].min();constraints=[];centers=[];sample=vertices[np.linspace(0,len(vertices)-1,min(1400,len(vertices))).astype(int)]
 if fixed_rotation is not None:rotation=np.asarray(fixed_rotation)
 support=vertices if support_vertices is None else support_vertices;floor_translation=.6505-(support@rotation.T)[:,2].min();height=floor_translation+float((rotation@center)[2])
 for i in anchor_indices:
  target=cv2.resize(cv2.imread(str(rec/f'masks/{object_index+1}/{i:06d}.png'),0),(512,384),interpolation=cv2.INTER_NEAREST)>0;valid=~g.human_occlusion(measure,i)
  for b in range(len(cfg['objects'])):
   if b!=object_index:valid&=cv2.resize(cv2.imread(str(rec/f'masks/{b+1}/{i:06d}.png'),0),(512,384),interpolation=cv2.INTER_NEAREST)==0
  yy,xx=np.where(target&valid)
  if len(xx)<30:continue
  ray=np.array([(xx.mean()*2-intr['cx'])/intr['fx'],(yy.mean()*2-intr['cy'])/intr['fy'],1.]);direction=cameras[i,:3,:3]@ray;origin=cameras[i,:3,3];centers.append(origin+direction*((height-origin[2])/direction[2]));take=np.linspace(0,len(xx)-1,min(90,len(xx))).astype(int);constraints.append((int(i),np.linalg.inv(cameras[i]),distance_transform_edt(~target),valid,np.c_[xx[take],yy[take]],target))
 assert constraints,'No clear resting-tool observations';guess=np.median(centers,axis=0)
 def pose(x):
  q=Rotation.from_rotvec([0.,0.,x[2]*.1]).as_matrix()@rotation;M=np.eye(4);M[:3,:3]=q;centroid=guess.copy();centroid[:2]+=x[:2]*.01;M[:3,3]=centroid-q@center;M[2,3]=floor_translation;return M
 def residual(x):
  model=pose(x);values=[]
  for i,C,dist,valid,pixels,target in constraints:
   camera=C@model;pc=sample@camera[:3,:3].T+camera[:3,3];uv=g.project(pc)*.5;clipped=np.c_[np.clip(uv[:,0],0,511),np.clip(uv[:,1],0,383)];visible=map_coordinates(valid.astype(float),clipped[:,::-1].T,order=0,mode='nearest');values.append((map_coordinates(dist,clipped[:,::-1].T,order=1,mode='nearest')+np.linalg.norm(uv-clipped,axis=1))*visible/2/np.sqrt(len(uv)/40));values.append(cKDTree(uv).query(pixels)[0]/2/np.sqrt(len(pixels)/20))
  return np.concatenate([*values,x[:2]/20,x[2:]/10])
 solution=least_squares(residual,np.zeros(3),bounds=([-10,-10,-10],[10,10,10]),loss='soft_l1',max_nfev=100);M=pose(solution.x);iou=[g.overlap(g.silhouette(vertices,C@M),target,valid)['iou'] for _,C,_,valid,_,target in constraints];return M,{'method':'Stationary support pose fitted to clear source silhouettes; fixed native support orientation' if fixed_rotation is not None else __doc__,'anchor_frames':[x[0] for x in constraints],'mean_anchor_iou':float(np.mean(iou)),'optimizer_evaluations':solution.nfev,'world_pose':M.tolist(),'scale_unchanged':True}
