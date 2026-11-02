"""Compare reconstructed robot FK with observed human landmarks over every original frame."""
from pathlib import Path
import argparse,importlib.util,json
import numpy as np,pyarrow.parquet as pq
parser=argparse.ArgumentParser();parser.add_argument('--run',type=Path,required=True);parser.add_argument('--reference',default='track3_collision_variant');parser.add_argument('--output',required=True);args=parser.parse_args();r=args.run;cfg=json.loads((r/'episode-config.json').read_text());spec=importlib.util.spec_from_file_location('retarget_source_geometry',r/'scripts/source_geometry.py');g=importlib.util.module_from_spec(spec);spec.loader.exec_module(g);measure=np.load(r/'reconstruction/hand_measurements.npz');camera=np.load(r/'reconstruction/scene_motion_physical.npz')['camera_to_world'];row=pq.ParquetFile(next((r/f'human_motion_data/{args.reference}').rglob('motion.parquet'))).read().to_pylist()[0];result=[];records=[]
for si,side in enumerate(['left','right']):
 names=row[f'{side}_robot_frame_names'];frames=np.array(row[f'robot_{side}_frames']);indices=[names.index(f'{side}_hand_C_MC')]
 for finger in ['thumb','index','middle','ring','pinky']:
  segments=['CMC_VL','MCP_VL','DP'] if finger=='thumb' else ['MCP_VL','MP','DP'];indices.extend([names.index(f'{side}_{finger}_{segment}') for segment in segments]+[names.index(f'{side}_{finger}_tip_site')])
 all_errors=[];free_errors=[];all_wrists=[];free_wrists=[]
 for i in range(cfg['frames']):
  if not measure['observed'][i,si]:continue
  C=np.linalg.inv(camera[i]);points=frames[i,indices,:3];uv=g.project(points@C[:3,:3].T+C[:3,3]);error=np.linalg.norm(uv-measure['uv'][i,si],axis=1);held=any(x['side']==side and x['start']<=i<=x['end'] for x in cfg['grasps']);all_errors.extend(error.tolist());all_wrists.append(float(error[0]));records.append({'source_frame':i,'side':side,'held':held,'mean_landmark_error_px':float(error.mean()),'wrist_error_px':float(error[0])})
  if not held:free_errors.extend(error.tolist());free_wrists.append(float(error[0]))
 result.append({'side':side,'all_mean_landmark_error_px':float(np.mean(all_errors)),'free_mean_landmark_error_px':float(np.mean(free_errors)) if free_errors else None,'all_mean_wrist_error_px':float(np.mean(all_wrists)),'free_mean_wrist_error_px':float(np.mean(free_wrists)) if free_wrists else None})
report={'method':__doc__,'reference':args.reference,'aggregate':result,'frames':records};Path(args.output).write_text(json.dumps(report,indent=2));print(cfg['episode'],result,flush=True)
