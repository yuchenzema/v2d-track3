"""Build an isolated, source-revalidated table-clearance candidate for a pilot."""
import argparse
import copy
import json
from pathlib import Path
import shutil
import numpy as np
import pyarrow.parquet as pq
import trimesh
from improvement_common import BASE,BATCH,save,utc,digest


def lift_along_source_rays(objects,cameras,lift):
    result=objects.copy()
    ray=objects[:,:,:3,3]-cameras[:,None,:3,3]
    assert np.isfinite(lift).all() and np.all(lift>=0),'Invalid clearance displacement'
    active=lift>0
    assert np.all(np.abs(ray[:,:,2][active])>.05),'Viewing ray cannot establish vertical clearance'
    scale=np.zeros_like(lift);scale[active]=lift[active]/ray[:,:,2][active]
    assert np.all(1+scale>0),'Clearance would cross the source camera'
    result[:,:,:3,3]+=ray*scale[:,:,None]
    return result


def build(training_root):
    from source_cad_alignment import human_mask,object_visibility,metrics,qualifies
    from source_mask_rasterizer import MeshSilhouette
    from source_marker_templates import load as load_templates
    from source_final_scene_audit import audit
    from experimental_training_admission import assess,POLICY
    training_root=Path(training_root)
    cfg=json.loads((training_root/'episode-config.json').read_text())
    approved=Path(cfg['approved_source_cad_run'])
    original=json.loads((approved/'research/cad-alignment-report.json').read_text())
    source_trajectory=approved/'reconstruction/cad-source-trajectory.npz'
    assert digest(source_trajectory)==original['trajectory_sha256']
    source=np.load(source_trajectory);motion={k:source[k].copy() for k in source.files}
    parquet=next((training_root/f"human_motion_data/{cfg['training_reference']}/processed").rglob('motion.parquet'))
    row=pq.ParquetFile(parquet).read().to_pylist()[0]
    delta=np.array(row['object_body_position'])-motion['object_to_world'][:,:,:3,3]
    assert np.max(np.abs(delta[:,:,:2]))<1e-6 and np.min(delta[:,:,2])>=-1e-6,'Not a vertical table projection'
    lift=np.maximum(delta[:,:,2],0)
    lift[lift>1e-6]+=.00025
    motion['object_to_world']=lift_along_source_rays(motion['object_to_world'],motion['camera_to_world'],lift)
    shift=np.linalg.norm(motion['object_to_world'][:,:,:3,3]-source['object_to_world'][:,:,:3,3],axis=2)
    assert shift.max()<.025,'Candidate exceeds the bounded reconstruction repair'
    root=approved.with_name(approved.name+'_source_ray_clearance_'+original['trajectory_sha256'][:12])
    assert not root.exists(),'Candidate already exists; preserve its evidence'
    for folder in ['reconstruction','research','videos','logs','evaluation']:(root/folder).mkdir(parents=True,exist_ok=True)
    for path in (approved/'reconstruction').iterdir():
        if path.name in ['cad-source-trajectory.npz','mesh_priors']:continue
        (root/'reconstruction'/path.name).symlink_to(path.resolve())
    shutil.copytree(approved/'reconstruction/mesh_priors',root/'reconstruction/mesh_priors')
    source_cfg=json.loads((approved/'episode-config.json').read_text())
    source_cfg.update(run=str(root),experimental_only=True,source_reference_repaired=True)
    save(root/'episode-config.json',source_cfg)
    trajectory=root/'reconstruction/cad-source-trajectory.npz'
    np.savez_compressed(trajectory,**motion)
    rec=root/'reconstruction';intr=json.loads((rec/'camera_intrinsics.json').read_text())
    hands=np.load(rec/'hand_measurements.npz');renderers={}
    for name in source_cfg['objects']:
        mesh=trimesh.load(rec/f'mesh_priors/{name}.ply',process=False)
        vertices,inverse=np.unique(mesh.vertices,axis=0,return_inverse=True)
        renderers[name]=MeshSilhouette(vertices,inverse[mesh.faces],intr)
    out=BATCH/'focused_episodes_20261028'
    templates=load_templates(out/'huggingface_audit_20261028/marker-templates.json',
        out/'source-marker-feature-review.json',root/'research/source-marker-template-review.json')
    records=copy.deepcopy(json.loads(Path(original['frames_report']).read_text()))
    final=audit(root,source_cfg,motion,renderers,records,hands,intr,templates,
        human_mask,object_visibility,metrics,qualifies)
    frames=root/'research/cad-frame-metrics.json';save(frames,records)
    final.update(trajectory_sha256=digest(trajectory),frames_report_sha256=digest(frames))
    final_path=root/'research/cad-final-scene-audit.json';save(final_path,final)
    report=dict(original,updated_utc=utc(),run=str(root),trajectory_sha256=digest(trajectory),
        frames_report=str(frames),frames_report_sha256=digest(frames),final_scene_audit=str(final_path),
        local_overlap_failed_object_frames=sum(not r['local_overlap_pass'] for r in records),
        unobservable_object_frames=sum(r['visible_pixels']<60 for r in records),
        physical_projection_applied=True,reference_kind='source_revalidated_physical_candidate',
        experimental_only=True,episode_done=False,policy_training_started=False,
        source_candidate_parent=str(approved),source_candidate_parent_trajectory_sha256=original['trajectory_sha256'],
        repair_method='Bounded translation along each original source-camera ray; preserve CAD scale, orientation, timestamps and source images; independently recheck every final overlay and sphere',
        repair_algorithm_sha256=digest(Path(__file__)),
        maximum_translation_m=float(shift.max()),overlay=None)
    save(root/'research/cad-alignment-report.json',report)
    admission=assess(report,json.loads(POLICY.read_text()))
    save(root/'research/experimental-admission.json',admission)
    if admission['admitted']:
        registry=out/'experimental-source-candidates.json'
        entries=json.loads(registry.read_text()) if registry.exists() else {}
        entries[str(cfg['episode'])]=str(root/'research/cad-alignment-report.json');save(registry,entries)
    assert digest(source_trajectory)==original['trajectory_sha256'],'Original CAD trajectory changed'
    print(json.dumps(dict(candidate=str(root),maximum_translation_m=float(shift.max()),admission=admission)),flush=True)
    return root,admission


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--run',type=Path,required=True)
    build(p.parse_args().run)
