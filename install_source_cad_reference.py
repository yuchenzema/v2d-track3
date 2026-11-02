"""Install an approved source-CAD trajectory into an isolated retargeting run."""
import json
from pathlib import Path
import shutil
import tempfile
import trimesh
from improvement_common import save,utc,digest


def install_source_measurements(root,approved,source):
    """Keep measurements inside /episode; host-only symlinks fail in Docker."""
    rec=Path(root)/'reconstruction';installed=[]
    for name in ['frames','masks','depth','depth_stereo','stereo_valid','calibrated.mp4',
                 'camera_intrinsics.json','hand_measurements.npz','table-camera-calibration.json',
                 'camera-frame-calibration.json','camera-frame-calibration.npz',
                 'scene-reconstruction.json','sparse_stereo_geometry.npz']:
        src=Path(approved)/'reconstruction'/name;dst=rec/name
        if not src.exists():src=Path(source)/'reconstruction'/name
        if src.exists() and (dst.is_symlink() or not dst.exists()):
            with tempfile.TemporaryDirectory(dir=rec,prefix='.source-measurement-') as temporary:
                staged=Path(temporary)/name
                if src.is_dir():shutil.copytree(src.resolve(),staged,symlinks=False)
                else:shutil.copy2(src.resolve(),staged)
                if dst.is_symlink():dst.unlink()
                staged.replace(dst)
            installed.append(name)
    return installed


def install(root,approved):
    root=Path(root);approved=Path(approved)
    report=json.loads((approved/'research/cad-alignment-report.json').read_text())
    cfg=json.loads((root/'episode-config.json').read_text())
    if cfg.get('experimental_training'):
        from experimental_training_admission import require
        admission=require(report,cfg)
        save(root/'research/experimental-source-admission.json',dict(admission,
            trajectory_sha256=report['trajectory_sha256'],frames_report_sha256=report['frames_report_sha256']))
    else:
        assert report['all_source_frames_checked'] and report['local_overlap_failed_object_frames']==0
    assert report['unobservable_object_frames']==0
    trajectory=approved/'reconstruction/cad-source-trajectory.npz'
    assert digest(trajectory)==report['trajectory_sha256']
    marker=root/'research/approved-source-cad-installed.json'
    cfg=json.loads((root/'episode-config.json').read_text());source_cfg=json.loads((approved/'episode-config.json').read_text())
    source=Path(source_cfg['source_reconstruction_run']);rec=root/'reconstruction'
    install_source_measurements(root,approved,source)
    if marker.exists():return
    assets=rec/'mesh_priors';assets.mkdir(exist_ok=True)
    for src in (source/'reconstruction/mesh_priors').iterdir():
        if src.is_file():shutil.copy2(src,assets/src.name)
    for name in cfg['objects']:
        for suffix in ['ply','glb']:
            src=approved/f'reconstruction/mesh_priors/{name}.{suffix}'
            shutil.copy2(src,assets/src.name)
        trimesh.load(assets/f'{name}.ply',process=False).export(assets/f'{name}.obj')
        src=source/f'reconstruction/mesh_priors/{name}.urdf'
        if src.exists():shutil.copy2(src,assets/src.name)
    shutil.copy2(trajectory,rec/'scene_motion_physical.npz')
    cfg['approved_source_cad_run']=str(approved);cfg['fps']=source_cfg['fps'];save(root/'episode-config.json',cfg)
    for stage in ['calibrate_and_measure_hands','fresh_segmentation_depth_and_stereo','fresh_slam_foundationpose_and_source_fit']:
        save(root/f'research/rebuild-{stage}-completed.json',dict(completed_utc=utc(),source_cad_validated=True,approved_source=str(approved)))
    save(marker,dict(updated_utc=utc(),approved_source=str(approved),trajectory_sha256=report['trajectory_sha256'],
                    original_source_trajectory_preserved=True,physics_contact_validation_still_required=True))
