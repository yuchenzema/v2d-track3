"""Assemble metric scene estimates and refine rigid geometry against every source frame."""
from pathlib import Path
import argparse,json,re,subprocess
BASE=Path('/home/yu_eli/v2d-track3');OLD=BASE/'runs/episode_000005_newton_20261015/scripts'
p=argparse.ArgumentParser();p.add_argument('--run',required=True);a=p.parse_args();root=Path(a.run);c=json.loads((root/'episode-config.json').read_text());names=c['objects'];T=c['frames'];rec=root/'reconstruction'
from stabilize_camera import stabilize
stabilize(root)
if c.get('camera_odometry_method')=='sift_pnp_metric':
 from global_hand_identity import run as assign_hands
 assign_hands(root)
s=(OLD/'reconstruct_scene.py').read_text().replace('T=288;names=[\'white_pot\',\'wooden_spoon\']',f'T={T};names={names!r}')
if c.get('camera_odometry_method')=='sift_pnp_metric':
 s=s.replace("REC/'hand_measurements.npz'","REC/'hand_measurements_anatomical.npz'")
 s=s.replace('stereo tabletop plane and circle; static-scene DROID yaw','FoundationStereo table anchor and background SIFT/PnP metric camera motion')
 s=s.replace('MediaPipe source landmarks, calibrated PnP and stereo depth on visible skin; missing observations interpolated','MediaPipe source landmarks with whole-trajectory anatomical handedness voting, calibrated PnP and stereo depth on visible skin; missing observations interpolated')
s=s.replace('n if n=="white_pot" else "wooden_spoon_geometric"','n')
s=s.replace("for s in range(2):\n wrist=np.median(world_hands[:9,s,0],axis=0);world_hands[:9,s]+=wrist-world_hands[:9,s,0,None]\n",'')
if c.get('frame_cache_format')=='jpg':s=s.replace('frames/{i:06d}.png','frames/{i:06d}.jpg')
s=s.replace('FoundationPose TensorRT pot; visible spoon head/end landmarks with calibrated stereo and tabletop roll prior; source-pixel/physical refinement follows','FoundationPose TensorRT for each object using official CAD or episode-specific video spoon mesh; source-pixel/physical refinement follows')
s=s.replace('T*.65','max(2,T*.35)')
(rec.parent/'scripts/reconstruct_scene.py').write_text(s)
if not (rec/'scene_motion.npz').exists():subprocess.run([str(root/'.venv-video/bin/python'),str(root/'scripts/reconstruct_scene.py')],check=True)
if c.get('source_only_reconstruction'):
 import shutil
 shutil.copy2(rec/'scene_motion.npz',rec/'scene_motion_physical.npz')
 (rec/'physical-refinement.json').write_text(json.dumps({'method':'Source-only image reconstruction; no floor projection, physical object editing or temporal smoothing before CAD validation','evaluation_ground_truth_used':False,'source_only_reconstruction':True},indent=2))
 raise SystemExit(0)
# The fitter makes no upright-pot or flat-spoon assumption: orientation is 6DoF.
import cv2,numpy as np,trimesh
from scipy.spatial.transform import Rotation
from scipy.signal import savgol_filter
m=np.load(rec/'scene_motion.npz');out={k:m[k].copy() for k in m.files};hands=np.load(rec/'hand_measurements.npz');intr=json.loads((rec/'camera_intrinsics.json').read_text());W,H=512,384;scale=np.array([intr['fx'],intr['fy']])*.5;center=np.array([intr['cx'],intr['cy']])*.5
meshes=[trimesh.load(rec/f'mesh_priors/{n}.ply',process=False) for n in names];rng=np.random.default_rng(42);verts=[]
for mesh in meshes:
 v=mesh.vertices
 if len(v)<6000:v=np.r_[v,mesh.sample(20000)]
 if len(v)>30000:v=v[rng.choice(len(v),30000,replace=False)]
 verts.append(v)
def raster(v,M):
 p=v@M[:3,:3].T+M[:3,3];u=np.rint(p[:,:2]/np.maximum(p[:,2:],.001)*scale+center).astype(int);ok=(p[:,2]>.05)&(u[:,0]>=0)&(u[:,0]<W)&(u[:,1]>=0)&(u[:,1]<H);im=np.zeros((H,W),np.uint8);im[u[ok,1],u[ok,0]]=255
 return cv2.morphologyEx(im,cv2.MORPH_CLOSE,np.ones((3,3),np.uint8))>0
def hand_occlusion(i):
 im=np.zeros((768,1024),np.uint8)
 for side in range(2):
  if not hands['observed'][i,side]:continue
  u=np.rint(hands['uv'][i,side]).astype(int);cv2.fillConvexPoly(im,cv2.convexHull(u[[0,1,5,9,13,17]]),255)
  for f in range(5):
   ids=[0]+list(range(1+4*f,5+4*f))
   for a,b in zip(ids[:-1],ids[1:]):cv2.line(im,tuple(u[a]),tuple(u[b]),255,14)
 return cv2.resize(im,(W,H),interpolation=cv2.INTER_NEAREST)>0
def iou(x,y,valid):return float((x&y&valid).sum()/max(1,((x|y)&valid).sum()))
records=[]
for i in range(T):
 C=np.linalg.inv(out['camera_to_world'][i]);masks=[cv2.resize(cv2.imread(str(rec/f'masks/{b+1}/{i:06d}.png'),0),(W,H),interpolation=cv2.INTER_NEAREST)>0 for b in range(len(names))];exclude=hand_occlusion(i)
 for b,name in enumerate(names):
  raw=out['object_to_world'][i,b].copy();seed=raw.copy();v=verts[b];valid=~exclude
  for j in range(len(names)):
   if j!=b:valid&=~masks[j]
  target=masks[b];raw_iou=iou(raster(v,C@raw),target,valid)
  # A generic support constraint prevents table penetration, but preserves
  # rotations, inversion, stacking and assembly rather than forcing Z-up.
  def evaluate(p):
   M=seed.copy();M[:3,:3]=Rotation.from_rotvec(p[3:]).as_matrix()@seed[:3,:3];M[:3,3]+=p[:3]
   minz=np.min((v@M[:3,:3].T+M[:3,3])[:,2]);inside=np.linalg.norm(M[:2,3])<.26
   if inside and minz<.6502:M[2,3]+=.6502-minz
   score=iou(raster(v,C@M),target,valid);penalty=.006*np.linalg.norm(p[:3])/.05+.002*np.linalg.norm(p[3:])/.15
   return score-penalty,M,score
  p=np.zeros(6);best=evaluate(p)
  for translation,rotation in [(.012,.08),(.006,.04),(.003,.02)]:
   for repeat in range(2):
    improved=False
    for dim in range(6):
     chosen=None
     for sign in [-1,1]:
      q=p.copy();q[dim]+=sign*(translation if dim<3 else rotation)
      if np.linalg.norm(q[:3])>.07 or np.linalg.norm(q[3:])>.4:continue
      result=evaluate(q)
      if result[0]>best[0]+1e-6:chosen=q;best=result
     if chosen is not None:p=chosen;improved=True
    if not improved:break
  out['object_to_world'][i,b]=best[1];records.append({'frame':i,'object':name,'raw_mask_iou':raw_iou,'refined_mask_iou':best[2],'translation_change_m':float(np.linalg.norm(raw[:3,3]-best[1][:3,3]))})
 if i%20==0:print('Generic source fit',i,[(r['object'],round(r['refined_mask_iou'],3)) for r in records[-len(names):]],flush=True)
# A short 3-frame translation filter reduces jitter while retaining pauses.
for b in range(len(names)):out['object_to_world'][:,b,:3,3]=savgol_filter(out['object_to_world'][:,b,:3,3],3,1,axis=0)
assert all(np.isfinite(v).all() for v in out.values())
np.savez_compressed(rec/'scene_motion_physical.npz',**out)
from pose_continuity import optimize
optimize(root)
report={'method':__doc__,'frame_count':T,'orientation_assumption':'Free 6DoF; no upright/inversion restriction','evaluation_ground_truth_used':False,'raw_iou':{n:float(np.mean([r['raw_mask_iou'] for r in records if r['object']==n])) for n in names},'refined_iou_before_temporal_filter':{n:float(np.mean([r['refined_mask_iou'] for r in records if r['object']==n])) for n in names},'frames':records}
(rec/'physical-refinement.json').write_text(json.dumps(report,indent=2));print(json.dumps({k:v for k,v in report.items() if k!='frames'},indent=2))
