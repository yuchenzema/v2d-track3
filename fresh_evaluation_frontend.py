"""Rebuild evaluation measurements using the frontend validated on public videos.

Historical object trajectories and policies are deliberately not copied. The
same serial controller owns the episode slot while these stages execute.
"""
from pathlib import Path
import json,os,subprocess,time
import cv2
from improvement_common import BASE,save,utc,digest
from episode_execution_guard import register_episode_child
from public_validation_pipeline import copy_measurement

PY=BASE/'runs/episode_000005_newton_20261015/.venv-video/bin/python'
MEASUREMENTS=['frames','calibrated.mp4','camera_intrinsics.json','video-preparation.json',
    'hand_measurements.npz','camera-frame-calibration.npz','camera-frame-calibration.json',
    'sparse_stereo_geometry.npz','depth_foundation_stereo','foundation-stereo-sequence.json']


def restore_source_frame_cache(root,cfg):
    """Restore pruned PNGs from original RGB, preserving the measured camera grid."""
    import numpy as np
    root=Path(root);rec=root/'reconstruction';folder=rec/'frames'
    prep=json.loads((rec/'video-preparation.json').read_text())
    intr=json.loads((rec/'camera_intrinsics.json').read_text())
    width,height=intr['width'],intr['height'];count=cfg['frames']
    valid=[]
    for frame in range(count):
        image=cv2.imread(str(folder/f'{frame:06d}.png')) if (folder/f'{frame:06d}.png').is_file() else None
        valid.append(image is not None and image.shape==(height,width,3))
    missing=[frame for frame,ok in enumerate(valid) if not ok]
    report=dict(updated_utc=utc(),frames=count,verified_source_frames=count,restored_frames=[],
                original_video_used_for_restoration=bool(missing),evaluation_ground_truth_used=False)
    if missing:
        video=Path(cfg['source'])
        if Path(prep['input_video']).resolve()!=video.resolve() or prep['frames']!=count:
            raise RuntimeError('Cached preparation is not bound to this original video/frame grid')
        calibration=BASE/'datasets/video_to_data_challenge/track_3'/prep['calibration_source']
        camera=json.loads(calibration.read_text())['cameras']['ego_cam_a']
        sensor_w,sensor_h=camera['resolution']
        K=np.asarray(camera['intrinsic_matrix'],dtype=np.float64)
        D=np.asarray(camera['distortion_coefficients'],dtype=np.float64)
        if D.size!=14 or prep['resolution']!=[width,height]:
            raise RuntimeError('Source restoration requires the recorded 14-parameter camera grid')
        new_K,_=cv2.getOptimalNewCameraMatrix(K,D,(sensor_w,sensor_h),prep['undistortion_alpha'],(sensor_w,sensor_h))
        upright=new_K.copy();upright[0,2]=sensor_w-1-new_K[0,2];upright[1,2]=sensor_h-1-new_K[1,2]
        upright[0]*=width/sensor_w;upright[1]*=height/sensor_h
        if not np.allclose([upright[0,0],upright[1,1],upright[0,2],upright[1,2]],
                           [intr['fx'],intr['fy'],intr['cx'],intr['cy']],rtol=0,atol=1e-6):
            raise RuntimeError('Source restoration would change the measured camera intrinsics')
        mx,my=cv2.initUndistortRectifyMap(K,D,None,new_K,(sensor_w,sensor_h),cv2.CV_32FC1)
        cap=cv2.VideoCapture(str(video));folder.mkdir(exist_ok=True)
        try:
            if (int(cap.get(cv2.CAP_PROP_FRAME_COUNT))!=count or
                    abs(cap.get(cv2.CAP_PROP_FPS)-cfg['fps'])>1e-6):
                raise RuntimeError('Original source video timing does not match the preparation grid')
            for frame in range(count):
                ok,image=cap.read()
                if not ok:raise RuntimeError(f'Original source video decode failed at frame {frame}')
                if valid[frame]:continue
                image=cv2.resize(cv2.rotate(cv2.remap(cv2.rotate(image,cv2.ROTATE_180),mx,my,cv2.INTER_LINEAR),cv2.ROTATE_180),(width,height))
                target=folder/f'{frame:06d}.png';temporary=folder/f'{frame:06d}.restore-{os.getpid()}.png'
                try:
                    if not cv2.imwrite(str(temporary),image):raise RuntimeError(f'Cannot restore source frame {frame}')
                    temporary.replace(target)
                finally:temporary.unlink(missing_ok=True)
                report['restored_frames'].append(frame)
                if len(report['restored_frames'])%50==0:print('Restoring source frames',len(report['restored_frames']),'/',len(missing),flush=True)
        finally:cap.release()
        report.update(source_video_sha256=digest(video),camera_calibration_sha256=digest(calibration),
                      method='Original source RGB; inverse stream rotation, factory undistortion, upright resize; no calibrated-video re-encoding')
    save(rec/'source-frame-cache.json',report)
    return report


def ensure_monocular_depth(root,cfg):
    """A retained last frame does not certify a pruned monocular depth cache."""
    import batch_worker as vision
    import numpy as np
    root=Path(root);rec=root/'reconstruction';moge=rec/'depth_moge';depth=rec/'depth'
    intr=json.loads((rec/'camera_intrinsics.json').read_text())
    def complete(folder):
        for frame in range(cfg['frames']):
            path=folder/f'{frame:06d}.png'
            image=cv2.imread(str(path),cv2.IMREAD_UNCHANGED) if path.is_file() else None
            if image is None or image.shape!=(intr['height'],intr['width']) or image.dtype!=np.uint16:return False
        return True
    # Before the first fusion, depth holds raw MoGe (possibly a read-only
    # historical symlink). Move its directory entry, never its old target.
    if not moge.exists() and depth.exists() and not (rec/'stereo-depth-alignment.json').exists():
        depth.rename(moge)
    # The metric fusion output must always belong to this new run.
    if depth.is_symlink():depth.unlink()
    depth.mkdir(exist_ok=True)
    regenerated=not complete(moge)
    if regenerated:
        if moge.is_symlink():moge.unlink()
        moge.mkdir(exist_ok=True);(rec/'moge_intrinsics').mkdir(exist_ok=True)
        command=json.loads((BASE/'runs/episode_000005_newton_20261015/research/moge-command.json').read_text())[1:]
        command=[value.replace(str(BASE/'runs/episode_000005_newton_20261015'),str(root)) for value in command]
        command[command.index('--depth_folder')+1]='/episode/reconstruction/depth_moge'
        command[command.index('--batch_size')+1]='1'
        vision.docker(command,root/'logs/moge-cache-rebuild.log')
        if not complete(moge):raise RuntimeError('MoGe regeneration did not produce every source-frame depth map')
    save(rec/'monocular-depth-cache.json',dict(updated_utc=utc(),frames=cfg['frames'],
        all_source_depth_frames_verified=True,regenerated=regenerated,
        historical_measurement_target_written=False,evaluation_ground_truth_used=False))


def prepare(source,episode):
    import shutil
    source=Path(source);old=json.loads((source/'episode-config.json').read_text())
    from own_geometry import required as own_required
    own=own_required(base=BASE)
    root=BASE/f'runs/evaluation_episode_{episode:06d}_frontend_20261029_v4'
    if own:root=root.with_name(root.name+'_own_geometry_20261102_v3')
    for name in ['reconstruction','research','logs','scripts','evaluation','videos']:(root/name).mkdir(parents=True,exist_ok=True)
    for name in ['.venv-video','models']:
        target=root/name
        if not target.exists() and not target.is_symlink():target.symlink_to(BASE/'runs/episode_000005_newton_20261015'/name)
    measurements=list(MEASUREMENTS)
    if own:
        cached=BASE/f'runs/evaluation_episode_{episode:06d}_frontend_20261029_v4'
        if (cached/'reconstruction/foundation-stereo-sequence.json').is_file():source=cached
        measurements+=['depth_moge','moge_intrinsics',
                       'sift-pnp-odometry.json','slam',
                       'table-camera-calibration.npz','table-camera-calibration.json','table-initial-plane.npz']
    for name in measurements:
        src=source/'reconstruction'/name;dst=root/'reconstruction'/name
        # A resumed CAD repair already points at this fresh frontend. Copying
        # its cache onto itself also attempts copystat on Docker-owned dirs.
        if src.resolve()==dst.resolve():continue
        # These caches can be rewritten by source-mask/depth/camera stages.
        # Real copies protect historical measurements from in-place writers.
        mutable=name in {'camera_intrinsics.json','camera-frame-calibration.npz','camera-frame-calibration.json',
                         'table-camera-calibration.npz','table-camera-calibration.json','table-initial-plane.npz','sparse_stereo_geometry.npz','slam'}
        copier=shutil.copy2 if own and mutable else copy_measurement
        if src.is_dir():shutil.copytree(src,dst,copy_function=copier,dirs_exist_ok=True)
        elif src.is_file() and not dst.exists():copier(src,dst)
    moge=source/'reconstruction/depth_moge';depth=root/'reconstruction/depth'
    if moge.is_dir() and not depth.exists() and not (root/'reconstruction/depth_moge').exists():depth.symlink_to(moge)
    video=BASE/f'datasets/video_to_data_challenge/track_3/evaluation/videos/chunk-000/observation.images.ego_cam_a/episode_{episode:06d}.mp4'
    cap=cv2.VideoCapture(str(video));frames=int(cap.get(cv2.CAP_PROP_FRAME_COUNT));fps=cap.get(cv2.CAP_PROP_FPS);cap.release()
    assert frames==old['frames'] and frames>0 and fps>0,'Evaluation source video frame grid differs from cached input measurements'
    cfg=dict(episode=episode,sequence=f'episode_{episode:06d}',run=str(root),frames=frames,objects=old['objects'],source=str(video),fps=fps,
        resolution=[2028,1520],frame_cache_format='png',dataset_split='evaluation',source_only_reconstruction=True,
        validated_video_pipeline=True,full_sequence_stereo_required=True,camera_odometry_method='sift_pnp_metric',
        photometric_refinement=True,static_joint_refinement=True,source_feature_pose_enabled=True,
        submission_mesh_face_budget=3600,grasps=[],support_intervals={},
        evaluation_ground_truth_used=False,purpose='Fresh source-video measurements, not historical object-pose replay')
    if own:cfg.update(own_geometry_required=True,geometry_source='evaluation_rgb_depth',
                      photometric_refinement=True,source_feature_pose_enabled=True)
    save(root/'episode-config.json',cfg)
    if own and not (root/'research/source-instance-initialization.json').is_file():
        from source_instance_prompts import prepare as prepare_instances
        prepare_instances(source,root,cfg)
    save(root/'research/frontend-input-provenance.json',dict(updated_utc=utc(),measurement_source=str(source),copied_measurement_names=MEASUREMENTS,
        old_object_poses_copied=False,old_scene_motion_copied=False,old_policies_copied=False,old_grasp_timing_copied=False,evaluation_pose_gt_used=False))
    return root,cfg


def run(source,episode,progress):
    root,cfg=prepare(source,episode)
    stages=[('prepare','video-preparation.json'),('frames','source-frame-cache.json'),('prompts','sam2_prompts.json'),
        ('metric','stereo-depth-alignment.json'),('masks','masks-reviewed.json'),
        ('stereo','foundation-stereo-sequence.json'),('odometry','sift-pnp-odometry.json'),('generic','scene_motion_physical.npz')]
    for stage,marker in stages:
        if stage!='frames' and (root/'reconstruction'/marker).exists():continue
        with (root/f'logs/frontend-{stage}.log').open('a') as log:
            child=subprocess.Popen([str(PY),'-u',str(BASE/'fresh_evaluation_frontend.py'),'--episode',str(episode),'--root',str(root),'--stage',stage],cwd=BASE,stdout=log,stderr=subprocess.STDOUT)
            register_episode_child(child.pid)
            while child.poll() is None:progress(root,stage);time.sleep(10)
            if child.returncode:raise RuntimeError(f'Evaluation {episode} frontend {stage} failed; see {root}/logs/frontend-{stage}.log')
    save(root/'research/frontend-completed.json',dict(updated_utc=utc(),source_frames=cfg['frames'],pose_quality_passed=False,
        next_stage='CAD overlay and every-frame source alignment; pipeline completion is still unverified'))
    return root


if __name__=='__main__':
    import argparse
    parser=argparse.ArgumentParser();parser.add_argument('--episode',type=int,required=True);parser.add_argument('--root',type=Path,required=True);parser.add_argument('--stage',required=True)
    args=parser.parse_args();root=args.root;cfg=json.loads((root/'episode-config.json').read_text())
    import batch_worker as vision
    import public_validation_pipeline as common
    # Share source-only prompt and mask utilities without its public-data setup
    # or the public-only video-path remapping wrapper.
    common.ROOT=root;common.EPISODE=args.episode;common.setup=lambda:cfg
    if args.stage=='prepare':vision.prepare(root,cfg)
    elif args.stage=='frames':restore_source_frame_cache(root,cfg)
    elif args.stage=='prompts':common.source_prompts()
    elif args.stage=='metric':ensure_monocular_depth(root,cfg);vision.metric(root,cfg)
    elif args.stage=='masks':common.reviewed_masks()
    elif args.stage=='stereo':
        from foundation_stereo_sequence import run as stereo
        stereo(root)
    elif args.stage=='odometry':
        from sift_metric_odometry import run as odometry
        odometry(root)
    elif args.stage=='generic':vision.generic_reconstruction(root,cfg)
    else:raise ValueError(args.stage)
