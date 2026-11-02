"""Bind training admission to independently saved source CAD alignment evidence."""
import json
from pathlib import Path
from improvement_common import BATCH,digest

def require_source_cad_alignment(episode,training_root):
    folder=BATCH/'focused_episodes_20261028'
    policy=folder/'source-first-policy.json'
    if not policy.exists():return
    policy=json.loads(policy.read_text())
    if not policy.get('enabled') or episode not in policy.get('episodes',[]):return
    status=folder/'source-cad-status.json'
    state=json.loads(status.read_text()) if status.exists() else {}
    report=state.get('results',{}).get(str(episode),{})
    assert report.get('all_source_frames_checked'),'CAD overlay audit missing; source reconstruction must precede training'
    cfgpath=Path(training_root)/'episode-config.json'
    cfg=json.loads(cfgpath.read_text()) if cfgpath.exists() else {}
    if cfg.get('experimental_training'):
        if cfg.get('approved_source_cad_run'):
            report=json.loads((Path(cfg['approved_source_cad_run'])/'research/cad-alignment-report.json').read_text())
        from experimental_training_admission import require
        require(report,cfg)
    else:
        assert report.get('local_overlap_failed_object_frames')==0,'CAD does not overlap source objects on every frame; training blocked'
    assert report.get('unobservable_object_frames')==0,'Hidden source object poses remain unverified; training blocked'
    root=Path(report['run'])
    from own_geometry import required as own_required,validate as validate_own
    if own_required(folder):
        assert not cfg.get('actor_transfer'),'Public-scan-trained actor lineage is ineligible for the own-geometry rerun'
        validate_own(root,training_root)
    assert digest(root/'reconstruction/cad-source-trajectory.npz')==report['trajectory_sha256'],'CAD trajectory evidence is stale'
    assert digest(Path(json.loads((root/'episode-config.json').read_text())['source']))==report['source_video_sha256'],'Source video changed'
    for name,expected in report.get('geometry_sha256',{}).items():
        assert digest(Path(training_root)/f'reconstruction/mesh_priors/{name}.ply')==expected,'Training CAD differs from the mesh validated in the human video'
    import numpy as np
    with np.load(root/'reconstruction/cad-source-trajectory.npz') as approved,np.load(Path(training_root)/'reconstruction/scene_motion_physical.npz') as training:
        for name in ['camera_to_world','object_to_world']:
            assert np.array_equal(approved[name],training[name]),'Training trajectory differs from source-aligned CAD; revalidate its overlay'
        assert report.get('final_scene_occlusion_rechecked') is True,'Final scene occlusion audit missing; training blocked'
        audit=json.loads(Path(report['final_scene_audit']).read_text())
        assert audit.get('all_object_frames_rechecked') is True,'Final scene audit did not recheck every object-frame'
        assert audit.get('trajectory_sha256')==report['trajectory_sha256'],'Final scene audit belongs to another trajectory'
        assert digest(Path(report['frames_report']))==report.get('frames_report_sha256')==audit.get('frames_report_sha256'),'Final scene frame metrics changed after validation'
        # Physics projections can edit the processed robot reference without
        # editing scene_motion_physical.npz. Check the actual learner input too.
        import pyarrow.parquet as pq
        from scipy.spatial.transform import Rotation
        cfg=json.loads((Path(training_root)/'episode-config.json').read_text())
        parquet=Path(training_root)/f"human_motion_data/{cfg['training_reference']}/processed/sequence_id={cfg['sequence']}/robot_name=sharpa_wave/motion.parquet"
        row=pq.ParquetFile(parquet).read().to_pylist()[0]
        positions=np.asarray(row['object_body_position']);quaternions=np.asarray(row['object_body_wxyz'])
        expected=approved['object_to_world']
        assert positions.shape==expected[:,:,:3,3].shape,'Training reference uses an unvalidated source frame grid'
        assert np.allclose(positions,expected[:,:,:3,3],atol=1e-6,rtol=0),'Physics retargeting moved objects away from source-aligned CAD'
        rotations=Rotation.from_quat(quaternions.reshape(-1,4),scalar_first=True).as_matrix().reshape(expected[:,:,:3,:3].shape)
        assert np.allclose(rotations,expected[:,:,:3,:3],atol=1e-5,rtol=0),'Physics retargeting rotated objects away from source-aligned CAD'
