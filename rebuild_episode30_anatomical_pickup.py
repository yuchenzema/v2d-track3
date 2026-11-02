"""Resolve brush pose ambiguity using measured hand approach and source pixels."""
import argparse,importlib.util,json,shutil
from pathlib import Path
import cv2,numpy as np,pyarrow as pa,pyarrow.parquet as pq,trimesh
from scipy.spatial.transform import Rotation
from improvement_common import save,utc

p=argparse.ArgumentParser();p.add_argument('--run',type=Path,required=True);args=p.parse_args()
r=args.run;assert not (r/'coherent-input-checksums.json').exists()
cfg=json.loads((r/'episode-config.json').read_text());assert cfg['episode']==30
rec=r/'reconstruction';sequence='sequence_id=episode_000030/robot_name=sharpa_wave/motion.parquet'
dst=r/'human_motion_data/track3_pickup_anatomical/processed'/sequence
marker=r/'research/anatomical-pickup-reference.json'
if marker.exists() and dst.exists():raise SystemExit(0)
base_path=r/'human_motion_data/track3_collision_variant/processed'/sequence
base=pq.read_table(base_path).to_pylist()[0]
video=pq.read_table(r/'human_motion_data/track3_video/processed'/sequence).to_pylist()[0]
T=cfg['frames'];position=np.array(base['object_body_position']);quat=np.array(base['object_body_wxyz'])
objects=np.tile(np.eye(4),(T,1,1));objects[:,:3,3]=position[:,0];objects[:,:3,:3]=Rotation.from_quat(quat[:,0],scalar_first=True).as_matrix()
human=np.tile(np.eye(4),(T,1,1));human[:,:3,3]=np.array(video['robot_right_wrist_position']);human[:,:3,:3]=Rotation.from_quat(video['robot_right_wrist_wxyz'],scalar_first=True).as_matrix()
# Use the source-tracked hand, never the displaced constant-grasp template.
anchor=127;binding=np.linalg.inv(objects[anchor])@human[anchor]
flip=np.diag([-1.,-1.,1.,1.]);options=np.stack([objects,objects@flip],axis=1)
spec=importlib.util.spec_from_file_location('pickup_source_geometry',r/'scripts/source_geometry.py');geo=importlib.util.module_from_spec(spec);spec.loader.exec_module(geo)
motion=np.load(rec/'scene_motion_physical.npz');hands=np.load(rec/'hand_measurements.npz')
vertices=np.asarray(trimesh.load(rec/'mesh_priors/mini_sweeper.ply',process=False).vertices)[::5]
costs=np.full((T,2),np.inf);records=[]
for i in range(T):
    target=cv2.resize(cv2.imread(str(rec/f'masks/1/{i:06d}.png'),0),(512,384),interpolation=cv2.INTER_NEAREST)>0
    rack=cv2.resize(cv2.imread(str(rec/f'masks/2/{i:06d}.png'),0),(512,384),interpolation=cv2.INTER_NEAREST)>0
    valid=~geo.human_occlusion(hands,i)&~rack;camera=np.linalg.inv(motion['camera_to_world'][i])
    overlaps=[geo.overlap(geo.silhouette(vertices,camera@options[i,k]),target,valid)['iou'] for k in range(2)]
    wrist_errors=[]
    for k in range(2):
        desired=options[i,k]@binding
        error=float(np.linalg.norm(desired[:3,3]-human[i,:3,3]));wrist_errors.append(error)
        if k and (target&valid).sum()>=30 and overlaps[k]<overlaps[0]-.01:continue
        angular=(Rotation.from_matrix(human[i,:3,:3]).inv()*Rotation.from_matrix(desired[:3,:3])).magnitude()
        costs[i,k]=(error/.03+.15*angular/.3 if 30<=i<=245 else .001*k)+2*(1-overlaps[k])
    records.append({'frame':i,'original_iou':overlaps[0],'alternative_iou':overlaps[1],'predicted_grasp_wrist_errors_m':wrist_errors})
values=costs.copy();parents=np.zeros((T,2),int)
for i in range(1,T):
    for k in range(2):
        turn=[(Rotation.from_matrix(options[i-1,j,:3,:3]).inv()*Rotation.from_matrix(options[i,k,:3,:3])).magnitude() for j in range(2)]
        cumulative=values[i-1]+.35*np.asarray(turn)
        parents[i,k]=int(np.argmin(cumulative));values[i,k]+=cumulative[parents[i,k]]
branch=np.zeros(T,int);branch[-1]=np.argmin(values[-1])
for i in range(T-1,0,-1):branch[i-1]=parents[i,branch[i]]
for i,k in enumerate(branch):
    quat[i,0]=Rotation.from_matrix(options[i,k,:3,:3]).as_quat(scalar_first=True)
    records[i]['selected_branch']=int(k)
base['object_body_wxyz']=quat.tolist()
for key in base:
    if key.startswith('robot_right_') or key.startswith('mano_right_'):base[key]=video[key]
dst.parent.mkdir(parents=True,exist_ok=True);pq.write_table(pa.Table.from_pylist([base]),dst,compression='zstd')
stage=r/'human_motion_data/track3_pickup_anatomical/reconstructed_stage';stage.mkdir(exist_ok=True)
shutil.copy2(r/'human_motion_data/track3_collision_variant/reconstructed_stage/episode_000030_support.usda',stage)
right_finger=np.array(video['robot_right_finger_joints'])[34]
save(r/'evaluation/anatomical-grasp-seed.json',{'best_candidates':{'mini_sweeper':{'side':'right','world_inward_direction':[0.,0.,0.],
     'inward_m':0.,'finger_joints':right_finger.tolist()}}})
save(marker,{'updated_utc':utc(),'method':__doc__,'source_frames_scored':T,'orientation_branch_changes':int(branch.sum()),
     'all_object_positions_preserved':bool(np.array_equal(position,np.array(base['object_body_position']))),
     'right_robot_reference_restored_from_source_hand_tracking':True,'policy_states_edited':False,'records':records})
print('Anatomical reference prepared; brush orientation branches changed:',int(branch.sum()),flush=True)
