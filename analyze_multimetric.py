"""Local diagnostic proxies only; never host/held-out scores or pose edits."""
from pathlib import Path
import hashlib,json,sys,time
import numpy as np
import pandas as pd
import trimesh
from scipy.spatial.transform import Rotation
import fast_simplification

BASE=Path('/home/yu_eli/v2d-track3')
sys.path.insert(0,str(BASE/'submission_kit_20261102'))
from v2dlb.track3_pose import validate_packed_meshes
from v2dlb.mesh_common import sample_mesh_surface
OUT=BASE/'submission_review_20261102/multimetric_analysis';OUT.mkdir(exist_ok=True)
STAGE=BASE/'runs/evaluation_batch_20261015/submission_staging'


def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def save(name,value):(OUT/name).write_text(json.dumps(value,indent=2))
def clean(vertices,faces):
    m=trimesh.Trimesh(vertices,faces,process=True)
    m.update_faces(m.nondegenerate_faces());m.remove_unreferenced_vertices()
    return m


def surface_error(source,candidate,seed):
    # Bidirectional exact point-to-triangle distances in the unchanged body
    # frame, rather than nearest sampled points or confidential geometry.
    a=sample_mesh_surface(source.vertices,source.faces,1200,seed)
    b=sample_mesh_surface(candidate.vertices,candidate.faces,1200,seed)
    da=trimesh.proximity.closest_point(candidate,a)[1]
    db=trimesh.proximity.closest_point(source,b)[1]
    v=np.r_[da,db]
    return dict(mean_mm=float(v.mean()*1000),p95_mm=float(np.quantile(v,.95)*1000),
                maximum_sampled_mm=float(v.max()*1000))


def run():
    readiness=json.loads((STAGE/'readiness.json').read_text())
    baseline_file=STAGE/'track3_original20_legacy.parquet'
    frame=pd.read_parquet(baseline_file);meshes=validate_packed_meshes(frame)
    motion_rows=[];mesh_rows=[];dedup={}
    for ep,item in readiness['episodes'].items():
        ep=int(ep);names=item['identity']['object_names']
        try:
            root=Path(item['recording'])
            while root.parent!=root and not (root/'reconstruction/scene_motion_physical.npz').is_file():root=root.parent
            reference=np.load(root/'reconstruction/scene_motion_physical.npz')['object_to_world']
            pose=pd.read_parquet(STAGE/f'results/episode_{ep:06d}.parquet')
            columns=['pos_x','pos_y','pos_z','quat_x','quat_y','quat_z','quat_w']
            actual=pose[columns].to_numpy().reshape(-1,len(names),7)
            assert len(actual)==len(reference),'Different source clock; do not resample diagnostics'
            rotation=Rotation.from_quat(actual[:,:,3:].reshape(-1,4)).as_matrix().reshape(*actual.shape[:2],3,3)
            centers=[];expected=[]
            for slot,name in enumerate(names):
                info=item['meshes'][name];v,f=meshes[ep,slot];m=clean(v,f);center=m.centroid
                a=actual[:,slot,:3]+np.einsum('tij,j->ti',rotation[:,slot],center)
                r=reference[:,slot,:3,3]+np.einsum('tij,j->ti',reference[:,slot,:3,:3],center)
                error=np.linalg.norm(a-r,axis=1)*100
                motion_rows.append(dict(episode=ep,object=name,mean_source_center_error_cm=float(error.mean()),
                    p95_source_center_error_cm=float(np.quantile(error,.95)),
                    expected_motion_extent_cm=float(np.linalg.norm(r-r[0],axis=1).max()*100),
                    actual_motion_extent_cm=float(np.linalg.norm(a-a[0],axis=1).max()*100),
                    frames=len(a),reference=str(root/'reconstruction/scene_motion_physical.npz'),
                    diagnostic_only=True,not_kaggle_per_episode_scores=True))
                centers.append(a);expected.append(r)
            if len(names)==2:
                # Relative separation removes the arbitrary world translation;
                # rotation symmetry is deliberately not treated as a host score.
                pair=np.linalg.norm((centers[0]-centers[1])-(expected[0]-expected[1]),axis=1)*100
                for row in motion_rows[-2:]:row['relative_separation_proxy_cm']=float(pair.mean())
        except Exception as error:motion_rows.append(dict(episode=ep,error=f'{type(error).__name__}: {error}'))
        for slot,name in enumerate(names):
            info=item['meshes'][name];path=Path(info['source'])
            assert sha(path)==info['sha256'],'Source mesh changed'
            v,f=meshes[ep,slot];baseline=clean(v,f);key=(sha(path),hashlib.sha256(v.tobytes()+f.tobytes()).hexdigest())
            if key in dedup:
                row=dict(dedup[key],episode=ep,object=name,shared_geometry=True);mesh_rows.append(row);continue
            source=trimesh.load(path,force='mesh',process=True);source.update_faces(source.nondegenerate_faces());source.remove_unreferenced_vertices()
            base_error=surface_error(source,baseline,19)
            best=baseline;best_error=base_error;best_agg=None;trials=[]
            if len(source.faces)>4096 or len(source.vertices)>4096:
                for agg in [3.,5.,9.]:
                    v2,f2=fast_simplification.simplify(source.vertices,source.faces,target_count=4096,agg=agg,preserve_border=True)
                    candidate=clean(v2,f2)
                    if len(candidate.faces)>4096 or len(candidate.vertices)>4096:continue
                    if source.is_watertight and not candidate.is_watertight:continue
                    if np.max(np.abs(candidate.bounds-source.bounds))>.0005:continue
                    error=surface_error(source,candidate,19);trials.append(dict(aggressiveness=agg,**error))
                    if error['mean_mm']<best_error['mean_mm']*.98 and error['p95_mm']<=best_error['p95_mm']:
                        best,best_error,best_agg=candidate,error,agg
            selected=None
            if best_agg is not None:
                target=OUT/f'mesh_candidates/episode_{ep:06d}/{name}.ply';target.parent.mkdir(parents=True,exist_ok=True);best.export(target);selected=str(target)
            row=dict(episode=ep,object=name,source=str(path),source_sha256=sha(path),
                source_faces=len(source.faces),submitted_faces=len(baseline.faces),baseline_surface_error=base_error,
                best_surface_error=best_error,selected_aggressiveness=best_agg,candidate=selected,
                trials=trials,body_coordinates_unchanged=True,no_source_or_submitted_file_modified=True)
            dedup[key]=row;mesh_rows.append(row)
            save('mesh-diagnostics.partial.json',mesh_rows)
            print('mesh',ep,name,'baseline mm',base_error['mean_mm'],'candidate mm',best_error['mean_mm'],flush=True)
        save('motion-diagnostics.partial.json',motion_rows)
    valid=[r for r in motion_rows if 'mean_source_center_error_cm' in r]
    valid.sort(key=lambda r:-r['mean_source_center_error_cm'])
    save('motion-diagnostics.json',valid);pd.DataFrame(valid).to_csv(OUT/'motion-diagnostics.csv',index=False)
    save('mesh-diagnostics.json',mesh_rows)
    baseline=json.loads((BASE/'submission_review_20261102/latest-kaggle-scores.json').read_text())
    save('improvement-targets.json',dict(baseline_kaggle_scores=baseline,
         direction=dict(add_auc='higher',spider_sr='higher',maniptrans_sr='higher',rpe_cm='lower',mppe_cm='lower',cd_o_cm='lower'),
         motion_priority=valid[:12],mesh_candidates=[r for r in mesh_rows if r['candidate']],
         proxy_is_not_heldout_validation=True,all_six_metrics_improved=False,
         no_posthoc_trajectory_edits=True,no_baseline_files_replaced=True))
    print('Finished local diagnostics:',len(valid),'object trajectories;',len(mesh_rows),'meshes;',sum(bool(r['candidate']) for r in mesh_rows),'mesh candidates',flush=True)


if __name__=='__main__':run()
