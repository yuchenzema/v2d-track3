"""Render source-image overlays for manual review of supported physical references."""
from pathlib import Path
import argparse,importlib.util,json
import cv2,numpy as np,pyarrow.parquet as pq,trimesh
from scipy.spatial.transform import Rotation

p=argparse.ArgumentParser();p.add_argument('--run',type=Path,required=True);p.add_argument('--reference');a=p.parse_args();r=a.run;c=json.loads((r/'episode-config.json').read_text());reference=a.reference or c['training_reference'];row=pq.ParquetFile(next((r/f'human_motion_data/{reference}/processed').rglob('motion.parquet'))).read().to_pylist()[0];motion=np.load(r/'reconstruction/scene_motion_physical.npz');h=np.load(r/'reconstruction/hand_measurements.npz');spec=importlib.util.spec_from_file_location('support_source_geometry',r/'scripts/source_geometry.py');g=importlib.util.module_from_spec(spec);spec.loader.exec_module(g);meshes=[trimesh.load(r/f'reconstruction/mesh_priors/{name}.ply',process=False) for name in c['objects']];P=np.array(row['object_body_position']);Q=np.array(row['object_body_wxyz']);panels=[]
frames={14:[80,96,108,116,130,157],15:[90,97,105,110,135,165],16:[94,97,104,110,144,172],19:[16,24,32,80,150,234],20:[132,145,159,167,185,201],25:[300,309,315,321,328,355],28:[120,145,165,190,205,219],30:[38,42,203,211,245,262],32:[87,92,98,150,155,160]}[c['episode']]
for i in frames:
 im=cv2.imread(str(r/f'reconstruction/frames/{i:06d}.jpg'));C=np.linalg.inv(motion['camera_to_world'][i])
 for b,mesh in enumerate(meshes):
  M=np.eye(4);M[:3,:3]=Rotation.from_quat(Q[i,b],scalar_first=True).as_matrix();M[:3,3]=P[i,b];mask=g.silhouette(mesh.vertices,C@M);contours,_=cv2.findContours((mask*255).astype('uint8'),cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_SIMPLE);cv2.drawContours(im,[v*2 for v in contours],-1,[(0,255,0),(255,80,255)][b],2)
 for si,side in enumerate(['left','right']):
  names=row[f'{side}_robot_frame_names'];indices=[names.index(f'{side}_hand_C_MC')]
  for finger in ['thumb','index','middle','ring','pinky']:
   segments=['CMC_VL','MCP_VL','DP'] if finger=='thumb' else ['MCP_VL','MP','DP'];indices.extend([names.index(f'{side}_{finger}_{seg}') for seg in segments]+[names.index(f'{side}_{finger}_tip_site')])
  pts=np.array(row[f'robot_{side}_frames'])[i,indices,:3];uv=np.rint(g.project(pts@C[:3,:3].T+C[:3,3])).astype(int)
  for finger in range(5):cv2.polylines(im,[uv[[0,*range(1+finger*4,5+finger*4)]]],False,(255,220,0) if si==0 else (0,120,255),2)
  if h['observed'][i,si]:
   for point in h['uv'][i,si].astype(int):cv2.circle(im,tuple(point),2,(0,255,255),-1)
 cv2.putText(im,f'Supported reference | source frame {i}',(12,32),0,.7,(255,255,255),2);panels.append(cv2.resize(im,(512,384)))
out=r/'research/supported-reference-visual-review.jpg';cv2.imwrite(str(out),np.concatenate([np.concatenate(panels[k:k+2],axis=1) for k in range(0,len(panels),2)]));print(out,flush=True)
