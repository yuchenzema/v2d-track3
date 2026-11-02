"""Compare every physically adjusted reference frame with original video pixels; no policy inference."""
from pathlib import Path
import argparse,json,importlib.util
import numpy as np,pyarrow.parquet as pq,trimesh,cv2
from scipy.spatial.transform import Rotation
parser=argparse.ArgumentParser();parser.add_argument('--run',required=True,type=Path);parser.add_argument('--reference',default='track3_collision_variant');parser.add_argument('--overlay',action='store_true');parser.add_argument('--output',default='physical-reference-source-fidelity');parser.add_argument('--no-update-selection',action='store_true');args=parser.parse_args();r=args.run.resolve();c=json.loads((r/'episode-config.json').read_text());spec=importlib.util.spec_from_file_location('video_source_geometry',r/'scripts/source_geometry.py');g=importlib.util.module_from_spec(spec);spec.loader.exec_module(g);row=pq.ParquetFile(next((r/f'human_motion_data/{args.reference}').rglob('motion.parquet'))).read().to_pylist()[0];P=np.array(row['object_body_position']);Q=np.array(row['object_body_wxyz']);motion=np.load(r/'reconstruction/scene_motion_physical.npz');measure=np.load(r/'reconstruction/hand_measurements.npz');meshes=[trimesh.load(r/f'reconstruction/mesh_priors/{n}.ply',process=False) for n in c['objects']];rows=[];images=[];source=cv2.VideoCapture(str(r/'reconstruction/calibrated.mp4'));T=c['frames'];deltas=[]
for i in range(T):
 C=np.linalg.inv(motion['camera_to_world'][i]);ok,image=source.read();assert ok;valid=~g.human_occlusion(measure,i);masks=[cv2.resize(cv2.imread(str(r/f'reconstruction/masks/{b+1}/{i:06d}.png'),0),(512,384),interpolation=cv2.INTER_NEAREST)>0 for b in range(len(c['objects']))];frame={'source_frame':i,'objects':{}}
 for b,name in enumerate(c['objects']):
  M=np.eye(4);M[:3,:3]=Rotation.from_quat(Q[i,b],scalar_first=True).as_matrix();M[:3,3]=P[i,b];render=g.silhouette(meshes[b].vertices,C@M);visible=valid.copy()
  for other in range(len(c['objects'])):
   if other!=b:visible&=~masks[other]
  score=g.overlap(render,masks[b],visible);score['translation_from_source_fit_m']=float(np.linalg.norm(P[i,b]-motion['object_to_world'][i,b,:3,3]));score['rotation_from_source_fit_rad']=float((Rotation.from_matrix(motion['object_to_world'][i,b,:3,:3]).inv()*Rotation.from_matrix(M[:3,:3])).magnitude());frame['objects'][name]=score
  if args.overlay:
   contours,_=cv2.findContours((render*255).astype('uint8'),cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_SIMPLE);cv2.drawContours(image,[a*2 for a in contours],-1,[(0,255,0),(255,80,255)][b],2)
 if args.overlay and (i%24==0 or i==T-1):
  cv2.putText(image,f'Physical reconstruction reference | frame {i}',(15,35),0,.7,(255,255,255),2);images.append(cv2.resize(image,(512,384)))
 rows.append(frame)
source.release();aggregate={n:{'mean_iou':float(np.mean([f['objects'][n]['iou'] for f in rows])),'mean_centroid_error_px':float(np.mean([f['objects'][n]['centroid_error_px'] for f in rows if f['objects'][n]['centroid_error_px'] is not None])),'maximum_translation_from_source_fit_m':max(f['objects'][n]['translation_from_source_fit_m'] for f in rows),'maximum_rotation_from_source_fit_rad':max(f['objects'][n]['rotation_from_source_fit_rad'] for f in rows)} for n in c['objects']};report={'method':__doc__,'reference':args.reference,'source_frames':T,'aggregate':aggregate,'frames':rows,'occlusion_limitation':'Hand landmarks approximate hand occlusion; forearms and all other human occlusions are not fully segmented.'};(r/f'evaluation/{args.output}.json').write_text(json.dumps(report,indent=2))
for j in range(0,len(images),4):cv2.imwrite(str(r/f'evaluation/physical-reference-review-{j:02d}.jpg'),np.concatenate(images[j:j+4]))
marker=r/'research/collision-reference-selection.json'
if marker.exists() and not args.no_update_selection:
 value=json.loads(marker.read_text());value['physical_reference_source_fidelity']=aggregate;marker.write_text(json.dumps(value,indent=2))
print(c['episode'],json.dumps(aggregate),flush=True)
