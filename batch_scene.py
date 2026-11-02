"""Video scene utilities shared across the remaining evaluation episodes."""
from pathlib import Path
import argparse,json,cv2,numpy as np,trimesh,xml.etree.ElementTree as ET
from scipy.spatial.transform import Rotation
BASE=Path('/home/yu_eli/v2d-track3');PUBLIC=BASE/'datasets/video_to_data_challenge/track_3/public'
p=argparse.ArgumentParser();p.add_argument('--run',required=True);p.add_argument('--stage',required=True,choices=['assets','plane']);a=p.parse_args();ROOT=Path(a.run);REC=ROOT/'reconstruction';cfg=json.loads((ROOT/'episode-config.json').read_text());names=cfg['objects'];ext=cfg.get('frame_cache_format','png');OUT=REC/'mesh_priors';OUT.mkdir(exist_ok=True)
def save(path,data):path.write_text(json.dumps(data,indent=2))
def assets():
 report={}
 for name in names:
  if name=='wooden_spoon':continue
  source=PUBLIC/f'urdf/{name}_rigid.urdf';tree=ET.parse(source);root=tree.getroot();done={}
  for meshnode in root.findall('.//mesh'):
   src=(source.parent/meshnode.attrib['filename']).resolve()
   if src not in done:
    mesh=trimesh.load(src,process=False).to_geometry();suffix='_collision' if 'collision' in src.stem else '';stem=name+suffix
    mesh.export(OUT/f'{stem}.obj')
    if not suffix:
     mesh.export(OUT/f'{name}.ply');mesh.export(OUT/f'{name}.glb')
    done[src]=stem
   meshnode.set('filename',f'/episode/reconstruction/mesh_priors/{done[src]}.obj')
  tree.write(OUT/f'{name}.urdf')
  mesh=trimesh.load(OUT/f'{name}.ply',process=False)
  report[name]={'source_urdf':str(source),'source_meshes':[str(x) for x in done],'bounds_m':mesh.bounds.tolist(),'mass_source':'Public URDF placeholder, not a measured mass','evaluation_ground_truth_used':False}
 save(REC/'asset-provenance.json',report)
def plane():
 depth_folder='depth_foundation_stereo' if cfg.get('camera_odometry_method')=='sift_pnp_metric' else 'depth'
 intr=json.loads((REC/'camera_intrinsics.json').read_text());im=cv2.imread(str(REC/f'frames/000000.{ext}'));hsv=cv2.cvtColor(im,cv2.COLOR_BGR2HSV);d=cv2.imread(str(REC/f'{depth_folder}/000000.png'),-1);z=65535/np.maximum(d.astype(float),1)-1
 yy,xx=np.indices(z.shape);valid=(hsv[:,:,0]>4)&(hsv[:,:,0]<28)&(hsv[:,:,1]>100)&(hsv[:,:,2]>65)&(yy>330)&(xx>250)&(xx<900)&(z>.2)&(z<1.4)
 for b in range(len(names)):valid&=cv2.dilate(cv2.imread(str(REC/f'masks/{b+1}/000000.png'),0),np.ones((21,21),np.uint8))==0
 if cfg.get('camera_odometry_method')=='sift_pnp_metric':
  from robust_table_camera_template import table_component
  hands=np.load(REC/'hand_measurements.npz')
  for side in range(2):
   if hands['observed'][0,side]:
    mask=np.zeros(valid.shape,np.uint8);cv2.fillConvexPoly(mask,cv2.convexHull(np.rint(hands['uv'][0,side]).astype(np.int32)),255);valid&=cv2.dilate(mask,np.ones((25,25),np.uint8))==0
  valid=table_component(valid)&(d>0)
 v,u=np.where(valid);rng=np.random.default_rng(42);ids=rng.choice(len(v),min(2500,len(v)),replace=False);v=v[ids];u=u[ids];zz=z[v,u];pc=np.c_[(u-intr['cx'])/intr['fx']*zz,(v-intr['cy'])/intr['fy']*zz,zz];best=np.zeros(len(pc),bool)
 for _ in range(400):
  x,y,z0=pc[rng.choice(len(pc),3,replace=False)];n=np.cross(y-x,z0-x);n/=np.linalg.norm(n)+1e-12;hit=np.abs((pc-x)@n)<.007
  if hit.sum()>best.sum():best=hit
 assert best.sum()>100
 center=pc[best].mean(0);_,_,V=np.linalg.svd(pc[best]-center,full_matrices=False);n=V[-1]
 if n@center>0:n=-n
 np.savez_compressed(REC/'table-initial-plane.npz',table_normal_camera=n,table_plane_d=float(n@center))
 save(REC/'table-initial-plane.json',{'inlier_fraction':float(best.mean()),'rms_m':float(np.sqrt(np.mean(((pc[best]-center)@n)**2))),'source':('Metric FoundationStereo on the connected tabletop, excluding hands and objects' if depth_folder=='depth_foundation_stereo' else 'Metric stereo/MoGE fusion on visible wood, excluding objects'),'evaluation_ground_truth_used':False})
if a.stage=='assets':assets()
else:plane()
