"""Rank complete saved rollouts using source-only diagnostics, never Kaggle labels."""
from pathlib import Path
import hashlib,json
import numpy as np
import trimesh
from scipy.spatial.transform import Rotation

BASE=Path('/home/yu_eli/v2d-track3')


def digest(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()


def run():
    ready=json.loads((BASE/'runs/evaluation_batch_20261015/submission_staging/readiness.json').read_text())
    rows=[];choices=[]
    for ep,item in ready['episodes'].items():
        ep=int(ep);names=item['identity']['object_names'];original=BASE/f'runs/episode_{ep:06d}_newton_batch_20261015'
        if not (original/'reconstruction/scene_motion_physical.npz').exists():continue
        ref=np.load(original/'reconstruction/scene_motion_physical.npz')['object_to_world']
        centers=[trimesh.load(item['meshes'][name]['source'],process=False).centroid for name in names]
        expected=np.array([ref[:,i,:3,3]+np.einsum('tij,j->ti',ref[:,i,:3,:3],center) for i,center in enumerate(centers)]).transpose(1,0,2)
        episode_rows=[]
        for root in (BASE/'runs').glob(f'episode_{ep:06d}_newton*'):
            path=root/'final-controller-selection.json'
            if not path.is_file():continue
            try:
                selected=json.loads(path.read_text())['selected']
                state=root/str(selected['recorded_states']).removeprefix('/episode/')
                checkpoint=root/str(selected['checkpoint']).removeprefix('/episode/')
                if not state.exists() or not checkpoint.exists():continue
                if not all((root/f'reconstruction/mesh_priors/{name}.ply').exists() and
                           digest(root/f'reconstruction/mesh_priors/{name}.ply')==item['meshes'][name]['sha256'] for name in names):continue
                with np.load(state,allow_pickle=False) as record:
                    if np.any(record['object_assistance']):continue
                    labels=record['body_names'].tolist();indices=[labels.index('track3_'+name+'/object') for name in names]
                    control=record['body_poses'][:,indices]
                    if len(control)!=2*(len(ref)-1):continue
                    if not np.array_equal(record['reference_frames'],np.arange(len(control))):continue
                    if not np.isfinite(control).all() or not np.allclose(np.linalg.norm(control[:,:,3:],axis=2),1,atol=1e-4):continue
                    poses=control[np.rint(np.arange(len(ref))*(len(control)-1)/(len(ref)-1)).astype(int)]
                actual=np.array([poses[:,i,:3]+Rotation.from_quat(poses[:,i,3:]).apply(center) for i,center in enumerate(centers)]).transpose(1,0,2)
                if len(names)==2:
                    a=actual[0,1]-actual[0,0];r=expected[0,1]-expected[0,0]
                    yaw=np.arctan2(r[1],r[0])-np.arctan2(a[1],a[0])
                else:yaw=0.
                rotation=Rotation.from_euler('z',yaw)
                aligned=rotation.apply(actual.reshape(-1,3)).reshape(actual.shape)
                aligned+=expected[0,0]-aligned[0,0]
                errors=np.linalg.norm(aligned-expected,axis=2)*100
                row=dict(episode=ep,objects=names,run=str(root),source_iou=selected.get('source_mean_object_iou'),
                    center_error_cm=errors.mean(axis=0).tolist(),p95_cm=np.quantile(errors,.95,axis=0).tolist(),
                    source_state_sha256=digest(state),checkpoint_sha256=digest(checkpoint),
                    recording=str(state),checkpoint=str(checkpoint),same_mesh_as_baseline=True,
                    object_assistance_zero=True,complete_source_sequence=True,diagnostic_only=True)
                episode_rows.append(row)
            except (KeyError,ValueError,OSError):continue
        baseline=next((r for r in episode_rows if r['source_state_sha256']==item['identity']['source_state_sha256']),None)
        if baseline:
            eligible=[r for r in episode_rows if r is not baseline and r.get('source_iou') is not None and
                baseline.get('source_iou') is not None and r['source_iou']>=baseline['source_iou'] and
                np.all(np.array(r['center_error_cm'])<=np.array(baseline['center_error_cm'])) and
                np.all(np.array(r['p95_cm'])<=np.array(baseline['p95_cm'])) and
                np.mean(r['center_error_cm'])<np.mean(baseline['center_error_cm'])*.9]
            if eligible:
                best=min(eligible,key=lambda r:np.mean(r['center_error_cm']))
                choices.append(dict(episode=ep,baseline=baseline,candidate=best))
        rows.extend(episode_rows)
        print('Compared source-only complete attempts:',ep,len(episode_rows),flush=True)
    result=dict(compared_attempts=rows,eligible_replacements=choices,
                no_posthoc_pose_edits=True,no_kaggle_per_episode_groundtruth=True,
                reference_is_source_estimate_not_confidential_groundtruth=True,
                diagnostic_alignment_never_exported=True,all_six_metrics_improved=False)
    output=BASE/'submission_review_20261102/all-saved-policy-comparison.json'
    output.write_text(json.dumps(result,indent=2))
    print('Source-only eligible replacements:',[(r['episode'],r['candidate']['run']) for r in choices],flush=True)


if __name__=='__main__':run()
