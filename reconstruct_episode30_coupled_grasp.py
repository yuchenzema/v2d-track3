"""Fit source brush silhouettes and observed hand pixels with one physical grasp."""
import argparse,importlib.util,json,shutil
from pathlib import Path
import cv2,numpy as np,pyarrow as pa,pyarrow.parquet as pq,trimesh
from scipy.optimize import minimize
from scipy.spatial.transform import Rotation
from scipy.ndimage import distance_transform_edt,map_coordinates
from improvement_common import save,utc

p=argparse.ArgumentParser();p.add_argument('--run',type=Path,required=True);p.add_argument('--grasp-label',default='coupled-source127-grasp');a=p.parse_args();r=a.run
assert not (r/'coherent-input-checksums.json').exists()
marker=r/'research/coupled-brush-reconstruction.json'
if marker.exists():raise SystemExit(0)
cfg=json.loads((r/'episode-config.json').read_text());rec=r/'reconstruction';T=263
suffix='processed/sequence_id=episode_000030/robot_name=sharpa_wave/motion.parquet'
data=pq.read_table(r/'human_motion_data/track3_collision_variant'/suffix).to_pylist()[0]
video=pq.read_table(r/'human_motion_data/track3_video'/suffix).to_pylist()[0]
fit=json.loads((r/f'evaluation/{a.grasp_label}.json').read_text())['best_candidates']['mini_sweeper']
assert fit['stable_hold_and_lift'];G=np.asarray(fit['wrist_in_object'])
spec=importlib.util.spec_from_file_location('coupled_source_geometry',r/'scripts/source_geometry.py');geo=importlib.util.module_from_spec(spec);spec.loader.exec_module(geo)
m=np.load(rec/'scene_motion_physical.npz');hands=np.load(rec/'hand_measurements.npz')
mesh=trimesh.load(rec/'mesh_priors/mini_sweeper.ply',process=False);vertices=np.asarray(mesh.vertices)[np.linspace(0,len(mesh.vertices)-1,6000).astype(int)]
position=np.array(data['object_body_position']);quats=np.array(data['object_body_wxyz'])
O=np.tile(np.eye(4),(T,1,1));O[:,:3,:3]=Rotation.from_quat(quats[:,0],scalar_first=True).as_matrix();O[:,:3,3]=position[:,0]
rack=np.tile(np.eye(4),(T,1,1));rack[:,:3,:3]=Rotation.from_quat(quats[:,1],scalar_first=True).as_matrix();rack[:,:3,3]=position[:,1]
human=np.tile(np.eye(4),(T,1,1));human[:,:3,:3]=Rotation.from_quat(video['robot_right_wrist_wxyz'],scalar_first=True).as_matrix();human[:,:3,3]=video['robot_right_wrist_position']
native_rest=O[0].copy();binding_rest=np.linalg.inv(rack[0])@native_rest
rest=np.asarray([rack[i]@binding_rest for i in range(T)])
flip=np.diag([-1.,-1.,1.,1.]);C34=np.linalg.inv(m['camera_to_world'][34])
uv34=hands['uv'][34,1,0]*.5
def wrist_uv(M,C):
    p=(C@M@G)[:3,3];return geo.project(p[None])[0]*.5
# Initial resting orientation is held consistently through contact and release.
if np.linalg.norm(wrist_uv(rest[34]@flip,C34)-uv34)<np.linalg.norm(wrist_uv(rest[34],C34)-uv34):rest=rest@flip
out=O.copy();records=[]
for i in range(T):
    C=np.linalg.inv(m['camera_to_world'][i]);target=cv2.resize(cv2.imread(str(rec/f'masks/1/{i:06d}.png'),0),(512,384),interpolation=cv2.INTER_NEAREST)>0
    other=cv2.resize(cv2.imread(str(rec/f'masks/2/{i:06d}.png'),0),(512,384),interpolation=cv2.INTER_NEAREST)>0
    valid=~geo.human_occlusion(hands,i)&~other;distance=distance_transform_edt(~target)
    wrist_target=hands['uv'][i,1,0]*.5
    def metrics(M):
        silhouette=geo.silhouette(vertices,C@M);iou=geo.overlap(silhouette,target,valid)['iou']
        wrist_error=float(np.linalg.norm(wrist_uv(M,C)-wrist_target))
        return iou,wrist_error
    original_iou,original_wrist=metrics(O[i])
    if i<=34 or i>=245:
        out[i]=rest[i];after,wrist=metrics(out[i])
    else:
        initial=[m['object_to_world'][i,0],m['object_to_world'][i,0]@flip,human[i]@np.linalg.inv(G),out[i-1]]
        if i>35:initial.append(human[i]@np.linalg.inv(human[i-1])@out[i-1])
        def objective(M):
            iou,err=metrics(M)
            points=vertices@M[:3,:3].T+M[:3,3];points=points@C[:3,:3].T+C[:3,3]
            uv=geo.project(points)*.5;uv=np.clip(uv,[0,0],[511,383]);d=float(np.mean(map_coordinates(distance,uv[:,::-1].T,order=1)))
            step=np.linalg.norm(M[:3,3]-out[i-1,:3,3]);turn=(Rotation.from_matrix(out[i-1,:3,:3]).inv()*Rotation.from_matrix(M[:3,:3])).magnitude()
            return 1-iou+.002*d+.08*(max(0.,err-6)/10)**2+.02*(step/.025)**2+.01*(turn/.25)**2
        seeds=[v.copy() for v in initial if np.linalg.norm(v[:3,3]-O[i,:3,3])<=.10]
        seed=min(seeds,key=objective)
        def transform(x):
            M=seed.copy();M[:3,3]+=x[:3];M[:3,:3]=Rotation.from_rotvec(x[3:]).as_matrix()@seed[:3,:3];return M
        result=minimize(lambda x:objective(transform(x)),np.zeros(6),method='Powell',bounds=[(-.035,.035)]*3+[(-.6,.6)]*3,
                        options={'maxiter':12,'maxfev':240,'xtol':.0002,'ftol':.0005})
        candidates=[seed,transform(result.x)]
        out[i]=min(candidates,key=objective);after,wrist=metrics(out[i])
    records.append({'frame':i,'source_iou_before':original_iou,'source_iou_after':after,
                    'grasp_wrist_pixel_error_before':original_wrist,'grasp_wrist_pixel_error_after':wrist,
                    'source_frame_observed':bool(hands['observed'][i,1])})
    if i%10==0:
        print('Coupled brush reconstruction source frame',i,'/',T,'IoU',after,'wrist error',wrist,flush=True)
        save(r/'research/coupled-reconstruction-progress.json',{'updated_utc':utc(),'source_frame':i,'total':T,'latest':records[-1]})
position[:,0]=out[:,:3,3];quats[:,0]=Rotation.from_matrix(out[:,:3,:3]).as_quat(scalar_first=True)
np.savez_compressed(rec/'brush_coupled_motion.npz',object_to_world=out,grasp_wrist_in_object=G)
data['object_body_position']=position.tolist();data['object_body_wxyz']=quats.tolist()
for key in data:
    if key.startswith('robot_right_') or key.startswith('mano_right_'):data[key]=video[key]
dest=r/'human_motion_data/track3_coupled_brush'/suffix;dest.parent.mkdir(parents=True,exist_ok=True);pq.write_table(pa.Table.from_pylist([data]),dest,compression='zstd')
stage=r/'human_motion_data/track3_coupled_brush/reconstructed_stage';stage.mkdir(exist_ok=True);shutil.copy2(r/'human_motion_data/track3_collision_variant/reconstructed_stage/episode_000030_support.usda',stage)
save(marker,{'updated_utc':utc(),'method':__doc__,'frames_scored':T,'fixed_physical_grasp':fit,'records':records,
     'mesh_scale_camera_and_source_clock_preserved':True,'policy_states_modified':False,'cold_pickup_validation_pending':True,
     'mean_source_iou_before':float(np.mean([v['source_iou_before'] for v in records])),
     'mean_source_iou_after':float(np.mean([v['source_iou_after'] for v in records]))})
