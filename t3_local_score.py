"""Public-only Track 3 diagnostics using unchanged official metric functions.

Evaluation ground truth is unavailable. Public surface samples, anchors and
symmetry axes are constructed from the released scans, not confidential assets.
Reconstruction-reference scores must never be labelled learned-policy scores.
"""
import argparse
import hashlib
import json
from pathlib import Path
import sys

import numpy as np
import pyarrow.parquet as pq
from scipy.spatial.transform import Rotation,Slerp
import trimesh

BASE=Path('/home/yu_eli/v2d-track3');KIT=BASE/'v2d_submission_kit'
sys.path.insert(0,str(KIT))
from v2dlb.track3_pose import _register_object_body,_convert_object_body,_align_registered_scene,_registered_pose_metric
from v2dlb.object_mesh import surface_moments,object_chamfer_cm
from v2dlb.mesh_common import _sample_mesh_surface

PUBLIC=BASE/'datasets/video_to_data_challenge/track_3/public'
METRICS={'AUC':'add_auc','SP-SR':'spider_sr','MP-SR':'maniptrans_sr','RPE_cm':'rpe_cm','MPPE_cm':'mppe_cm'}

def load_public(episode):
    path=PUBLIC/f'data/chunk-000/episode_{episode:06d}.parquet'
    rows=pq.read_table(path).to_pylist();names=[o['name'] for o in rows[0]['observation.objects']]
    poses=np.array([[o['pose'] for o in row['observation.objects']] for row in rows],dtype=float)
    valid=np.array([[o['visible'] for o in row['observation.objects']] for row in rows],bool)
    # Official pose scoring needs complete scene tensors. Hidden GT is excluded
    # jointly, rather than interpolating a zero quaternion into an invented pose.
    valid &= np.isfinite(poses).all(axis=-1)&(np.linalg.norm(poses[:,:,3:],axis=-1)>.5)
    poses=poses[:,:,[0,1,2,4,5,6,3]]
    return names,poses,valid,rows


def mesh_for(name):
    paths=[PUBLIC/f'mesh/{name}/{name}_visual.glb',PUBLIC/f'mesh/{name}/{name}.glb']
    path=next(p for p in paths if p.exists());m=trimesh.load(path,force='scene',process=False).to_geometry()
    return (np.asarray(m.vertices,dtype=float),np.asarray(m.faces,dtype=np.int64)),path


def symmetry(name,mesh):
    if name not in ['beige_cup','blue_cup','white_pot_lid']:return np.zeros(3),1
    points=mesh[0];center=points.mean(0);eigen,axes=np.linalg.eigh(np.cov((points-center).T))
    if name=='white_pot_lid':index=0;order=2
    else:
        ratios=[abs(eigen[(i+1)%3]-eigen[(i+2)%3])/max(eigen[(i+1)%3]+eigen[(i+2)%3],1e-12) for i in range(3)]
        index=int(np.argmin(ratios));order=360
    return axes[:,index],order


def shift_poses(poses,advance):
    if advance==0:return poses.copy()
    N=len(poses);time=np.clip(np.arange(N)+advance,0,N-1);out=poses.copy()
    for body in range(poses.shape[1]):
        for dim in range(3):out[:,body,dim]=np.interp(time,np.arange(N),poses[:,body,dim])
        out[:,body,3:]=Slerp(np.arange(N),Rotation.from_quat(poses[:,body,3:]))(time).as_quat()
    return out


def prediction_body_order(prediction,data,expected):
    """Resolve recorded array indices by names, never by Parquet row order."""
    if 'object_names' in data:
        names=[item.decode() if isinstance(item,bytes) else str(item) for item in data['object_names']]
        source='npz.object_names'
    else:
        config=Path(prediction).parent.parent/'episode-config.json'
        if not config.is_file():raise ValueError('Prediction object names missing; require object_names or episode-config.json')
        names=json.loads(config.read_text())['objects'];source=str(config)
    if len(names)!=len(expected) or len(set(names))!=len(names) or set(names)!=set(expected):
        raise ValueError(f'Prediction object roster differs from public objects: {names} versus {expected}')
    return [names.index(name) for name in expected],names,source


def compute(episode,prediction=None,mesh_dir=None,advance=0.,sample_count=1000,include_mesh=True):
    names,truth,valid,rows=load_public(episode);count=len(truth)
    input_names=None;order_source=None
    if prediction is None:estimated=truth.copy();kind='public_ground_truth_identity_control'
    elif Path(prediction).suffix=='.parquet':
        table=pq.read_table(prediction);metadata=table.schema.metadata or {}
        assert metadata.get(b'flash_chord.world_count')==b'1','Score the complete recorded single-world attempt, not a selected world from a cohort'
        assert metadata.get(b'flash_chord.eval_script_sha256') and metadata.get(b'flash_chord.checkpoint_sha256'),'Policy evaluator provenance missing'
        assert metadata.get(b'flash_chord.start_frame')==b'0'
        records=table.to_pylist();estimated=np.full(truth.shape,np.nan)
        seen=set();body_names={}
        for row in records:
            frame=int(row['reference_frame']);recorded_body=int(row['object_body_id'])
            assert 0<=frame<count and 0<=recorded_body<len(names),'Policy trajectory must preserve the source frame grid'
            actual=row['object_body_name'].removeprefix('track3_').split('/')[0]
            assert actual in names,'Unknown policy object name'
            assert body_names.setdefault(recorded_body,actual)==actual,'Policy object ID changes name within one recording'
            assert len(set(body_names.values()))==len(body_names),'Duplicate policy object names'
            body=names.index(actual)
            assert (frame,body) not in seen and row['environment_valid']
            seen.add((frame,body))
            estimated[frame,body]=[row['achieved_'+field] for field in ['position_x','position_y','position_z','quaternion_x','quaternion_y','quaternion_z','quaternion_w']]
        input_names=[body_names[body] for body in range(len(names))];order_source='parquet.object_body_name'
        kind='official_policy_rollout_not_kaggle_score'
    else:
        with np.load(prediction,allow_pickle=False) as data:
            order,input_names,order_source=prediction_body_order(prediction,data,names)
            if 'object_to_world' in data:
                matrices=data['object_to_world'];assert len(matrices)==count,'Source frame count differs from public GT'
                assert matrices.shape==(count,len(names),4,4),'Source object matrix shape differs from named roster'
                matrices=matrices[:,order]
                estimated=np.concatenate([matrices[:,:,:3,3],Rotation.from_matrix(matrices[:,:,:3,:3].reshape(-1,3,3)).as_quat().reshape(count,len(names),4)],axis=-1)
            elif 'poses_xyzw' in data:estimated=data['poses_xyzw'][:,order].copy()
            else:raise ValueError('Expected source object_to_world or poses_xyzw; do not use unlabelled trajectory data')
        kind='video_reconstructed_reference_not_policy'
    assert estimated.shape==truth.shape and np.isfinite(estimated).all()
    # Interpolate candidate poses only. Never shift GT or choose evaluation data
    # to tune the offset. Frame-order visibility masks stay on their source grid.
    if advance:estimated=shift_poses(estimated,advance)
    frame_valid=valid.all(axis=1)&np.isfinite(estimated).all(axis=(1,2))
    assert frame_valid[0],'First-frame scene alignment unavailable; do not silently use another frame'
    a=estimated[frame_valid,None].copy();r=truth[frame_valid,None].copy()
    samples=[];axes=[];orders=[];cd={};geometry=[]
    for body,name in enumerate(names):
        reference,path=mesh_for(name);candidate=reference;candidate_path=path
        if mesh_dir:
            candidate_path=next(p for p in [Path(mesh_dir)/f'{name}.ply',Path(mesh_dir)/f'{name}.glb'] if p.exists())
            m=trimesh.load(candidate_path,force='scene',process=False).to_geometry();candidate=(m.vertices,m.faces)
        anchor=surface_moments(*reference)[0];seed=1000003*episode+101*body
        if candidate_path==path:ca,rotation=anchor,np.eye(3)
        else:ca,rotation=_register_object_body(candidate,reference,anchor,seed)
        a[:,:,body]=_convert_object_body(a[:,:,body],ca,rotation)
        r[:,:,body,:3]+=np.einsum('...ij,j->...i',Rotation.from_quat(r[:,:,body,3:].reshape(-1,4)).as_matrix().reshape(-1,1,3,3),anchor)
        samples.append(_sample_mesh_surface(*reference,sample_count,seed)-anchor)
        axis,order=symmetry(name,reference);axes.append(axis);orders.append(order)
        if include_mesh:cd[name]=object_chamfer_cm(reference,candidate,seed)
        geometry.append(dict(object=name,reference_mesh=str(path),candidate_mesh=str(candidate_path),
                             candidate_mesh_sha256=hashlib.sha256(candidate_path.read_bytes()).hexdigest(),
                             symmetry_axis=axis.tolist(),symmetry_order=order))
    a=_align_registered_scene(a,r,axes)
    values={name:float(_registered_pose_metric(metric,a,r,samples,axes,orders)) for name,metric in METRICS.items()}
    per_object={body_name:{metric:float(_registered_pose_metric(implementation,a[:,:,body:body+1],r[:,:,body:body+1],samples[body:body+1],axes[body:body+1],orders[body:body+1]))
                           for metric,implementation in METRICS.items() if metric!='RPE_cm'}
                for body,body_name in enumerate(names)}
    if include_mesh:values['CD-O_cm']=float(np.mean(list(cd.values())))
    files=[KIT/'v2dlb/track3_pose.py',KIT/'v2dlb/object_mesh.py',KIT/'v2dlb/object_pose.py']
    return dict(episode=episode,kind=kind,metrics=values,per_object=per_object,
        prediction=str(prediction) if prediction else None,
        prediction_object_order=input_names,scoring_object_order=names,prediction_body_order_source=order_source,
        local_scorer_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        prediction_sha256=hashlib.sha256(Path(prediction).read_bytes()).hexdigest() if prediction else None,
        source_frames=count,frames_scored=int(frame_valid.sum()),
        excluded_invisible_frames=np.flatnonzero(~frame_valid).tolist(),candidate_advance_frames=advance,
        public_surface_sample_count=sample_count,geometry=geometry,mesh_chamfer_cm=cd,
        official_function_hashes={str(p.relative_to(KIT)):hashlib.sha256(p.read_bytes()).hexdigest() for p in files},
        exact_kaggle_score=False,evaluation_ground_truth_used=False,
        limitations=['Public mesh surface samples, centroid anchors and PCA symmetry axes replace confidential host geometry.',
                     'Invisible public GT frames are excluded, not filled. A reconstructed reference score is not a policy rollout score.'])


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--public-episode',type=int,required=True);p.add_argument('--prediction',type=Path);p.add_argument('--mesh-dir',type=Path);p.add_argument('--advance',type=float,default=0.);p.add_argument('--samples',type=int,default=1000);p.add_argument('--output',type=Path,required=True);p.add_argument('--skip-mesh',action='store_true');a=p.parse_args()
    report=compute(a.public_episode,a.prediction,a.mesh_dir,a.advance,a.samples,not a.skip_mesh)
    a.output.parent.mkdir(parents=True,exist_ok=True);a.output.write_text(json.dumps(report,indent=2)+'\n');print(json.dumps(report['metrics']),flush=True)
