"""Compare every source frame/object for each physical grip diagnostic world."""
from pathlib import Path
import os
for variable in ['OPENBLAS_NUM_THREADS','OMP_NUM_THREADS','MKL_NUM_THREADS']:
    os.environ[variable]='1'
import argparse
import importlib.util
import json
import cv2
import numpy as np
import trimesh
from scipy.spatial.transform import Rotation
from improvement_common import BASE,BATCH,baseline,root,digest,save,utc
from object_motion_quality import summarize,nonregressing_objects


def assess(ep):
    cv2.setNumThreads(1)
    directory=BATCH/'improvements_20261020_grasp'
    status=json.loads((directory/f'episode_{ep:06d}-grip-sweep-status.json').read_text())
    if status['status']!='completed':raise ValueError('Wait for the physical grip diagnostic')
    run=BASE/'runs'/f'episode_{ep:06d}_newton_grasp_curriculum_20261020'
    cfg=json.loads((run/'episode-config.json').read_text())
    diagnostic=Path(status['diagnostic']);details=json.loads(diagnostic.read_text())
    data=np.load(diagnostic.with_suffix('.npz'));poses=data['achieved_pose_w'];expected=data['reference_pose_w']
    assert data['object_body_names'].tolist()==cfg['objects']
    assert len(poses)==2*(cfg['frames']-1)
    original=baseline(ep)
    stem=Path(original['recorded_states']).stem.removesuffix('-states')
    phases=json.loads((root(ep)/f'evaluation/{stem}-source-fidelity.json').read_text())['phases']
    spec=importlib.util.spec_from_file_location('grip_source_geometry',run/'scripts/source_geometry.py')
    geometry=importlib.util.module_from_spec(spec);spec.loader.exec_module(geometry)
    motion=np.load(run/'reconstruction/scene_motion_physical.npz')
    measurement=np.load(run/'reconstruction/hand_measurements.npz')
    meshes=[trimesh.load(run/f'reconstruction/mesh_priors/{name}.ply',process=False) for name in cfg['objects']]
    sample=np.rint(np.arange(cfg['frames'])*(len(poses)-1)/(cfg['frames']-1)).astype(int)
    worlds=[{'frames':[],'phases':phases} for _ in details['profiles']]
    for frame in range(cfg['frames']):
        camera=np.linalg.inv(motion['camera_to_world'][frame])
        visible=~geometry.human_occlusion(measurement,frame)
        masks=[cv2.resize(cv2.imread(str(run/f'reconstruction/masks/{b+1}/{frame:06d}.png'),0),
            (512,384),interpolation=cv2.INTER_NEAREST)>0 for b in range(len(cfg['objects']))]
        for world in range(len(worlds)):
            objects={}
            for b,name in enumerate(cfg['objects']):
                valid=visible.copy()
                for other,mask in enumerate(masks):
                    if other!=b:valid&=~mask
                render=geometry.silhouette(meshes[b].vertices,camera@geometry.matrix(poses[sample[frame],world,b]))
                objects[name]=geometry.overlap(render,masks[b],valid)
            worlds[world]['frames'].append({'source_frame':frame,'objects':objects})
    rows=[]
    for world,record in enumerate(details['profiles']):
        per_body={}
        for b,name in enumerate(cfg['objects']):
            distance=np.linalg.norm(poses[:,world,b,:3]-expected[:,b,:3],axis=1)
            angle=(Rotation.from_quat(expected[:,b,3:]).inv()*Rotation.from_quat(poses[:,world,b,3:])).magnitude()
            per_body[name]={'mean_position_error_m':float(distance.mean()),'max_position_error_m':float(distance.max()),
                'mean_orientation_error_rad':float(angle.mean()),'max_orientation_error_rad':float(angle.max())}
        tracking={'passed_all_frames':record['passed_all_tracking_frames'],'per_body':per_body}
        quality=summarize(cfg,worlds[world],tracking)
        rows.append({**record,'object_motion_quality':quality,
            'passes_initial_per_object_nonregression':nonregressing_objects({'object_motion_quality':quality},original)})
    result={'episode':ep,'updated_utc':utc(),'diagnostic_only':True,
        'source_frames_checked_per_profile':cfg['frames'],'source_sha256':digest(cfg['source']),
        'physical_diagnostic':str(diagnostic),'physical_states_sha256':digest(diagnostic.with_suffix('.npz')),
        'profiles':rows,'full_32_world_validation_still_required':True,
        'full_source_frame_metrics_per_profile':worlds}
    save(directory/f'episode_{ep:06d}-grip-sweep-source-review.json',result)
    for row in sorted(rows,key=lambda r:r['object_motion_quality']['score'],reverse=True)[:5]:
        print(ep,row['profile'],row['passes_initial_per_object_nonregression'],round(row['object_motion_quality']['score'],3),
            {n:round(v['mean_iou'],3) for n,v in row['object_motion_quality']['objects'].items()},flush=True)
    return result


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('episode',type=int)
    assess(parser.parse_args().episode)
