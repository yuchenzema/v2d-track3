"""Compare genuine policy recordings against shared measured source images."""
import json
from pathlib import Path
import numpy as np,trimesh,pyarrow.parquet as pq
from scipy.spatial.transform import Rotation
from source_mask_rasterizer import MeshSilhouette
from own_source_quality import observation,image_cost


def score(root,trajectory,source):
    root,source=Path(root),Path(source);cfg=json.loads((root/'episode-config.json').read_text());rec=source/'reconstruction'
    intr=json.loads((rec/'camera_intrinsics.json').read_text());camera=np.load(rec/'scene_motion_physical.npz')['camera_to_world']
    table=pq.read_table(trajectory).to_pandas();scores=[];ious=[];depths=[]
    frames=np.linspace(0,cfg['frames']-1,min(40,cfg['frames']),dtype=int)
    for slot,name in enumerate(cfg['objects']):
        mesh=trimesh.load(root/f'reconstruction/mesh_priors/{name}.ply',force='mesh',process=False);renderer=MeshSilhouette(mesh.vertices,mesh.faces,intr)
        rows=table[table.object_id==slot].sort_values('step')
        if len(rows)!=cfg['frames']:
            ids=sorted(table.object_id.unique());rows=table[table.object_id==ids[slot]].sort_values('step')
        position=rows[[f'achieved_position_{axis}' for axis in 'xyz']].to_numpy();quaternion=rows[[f'achieved_quaternion_{axis}' for axis in 'xyzw']].to_numpy()
        rotation=Rotation.from_quat(quaternion).as_matrix()
        for frame in frames:
            pose=np.eye(4);pose[:3,:3]=rotation[frame];pose[:3,3]=position[frame];pose=np.linalg.inv(camera[frame])@pose
            mask,depth,valid=observation(rec,slot+1,int(frame));cost,metrics=image_cost(renderer,pose,mask,depth,valid)
            scores.append(cost);ious.append(metrics['iou']);depths.append(metrics['median_depth_error_m'])
    return dict(source_video_loss=float(np.mean(scores)),mean_source_mask_iou=float(np.mean(ious)),median_source_depth_error_m=float(np.median(depths)),
        inspected_object_frames=len(scores),strict_episode_completion=False,not_kaggle_score=True)
