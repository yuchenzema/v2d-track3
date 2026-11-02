"""Resumeable local stages; each episode keeps its own inputs and logs."""
from pathlib import Path
import argparse, concurrent.futures, json, os, re, shlex, shutil, subprocess, time
BASE=Path('/home/yu_eli/v2d-track3');PRIOR=BASE/'runs/episode_000005_newton_20261015';MODULES=BASE/'video_to_data/reconstruction/modules';WEIGHTS=BASE/'video_to_data/reconstruction/data/weights'
PY=str(PRIOR/'.venv-video/bin/python')
def run(args,log):
    log.parent.mkdir(parents=True,exist_ok=True)
    log.with_suffix('.command.json').write_text(json.dumps(args,indent=2))
    with log.open('w') as f:subprocess.run(args,stdout=f,stderr=subprocess.STDOUT,check=True)
def docker(args,log):
    from batch_gpu_gate import auxiliary_gpu
    with auxiliary_gpu():run(['sg','docker','-c',shlex.join(['docker',*args])],log)
def context(ep):
    root=BASE/'runs'/f'episode_{ep:06d}_newton_batch_20261015';cfg=json.loads((root/'episode-config.json').read_text());return root,cfg
def prepare(root,cfg):
    if (root/'reconstruction/video-preparation.json').exists():return
    s=(PRIOR/'scripts/prepare_video.py').read_text().replace('episode_000005',cfg['sequence'])
    s=s.replace('import mediapipe as mp','import mediapipe as mp\nfrom scipy.optimize import linear_sum_assignment')
    start=s.index('        for candidate_index, (image_points, world_points, handedness)')
    stop=s.index('            if category.score < scores[index,side]:',start)
    s=s[:start]+'''        # Match full landmark sets to their own recent tracks, so identity is
        # retained when the hands cross during an observed object handover.
        assignment_cost = np.zeros((len(candidates),2))
        for ci,(ip,wp,hc) in enumerate(candidates):
            current=np.array([[p.x*target_w,p.y*target_h] for p in ip])
            for side in range(2):
                recent=np.flatnonzero(observed[:index,side])
                if len(recent) and index-recent[-1]<=20:
                    predicted=uv[recent[-1],side].copy()
                    if len(recent)>1:
                        velocity=(uv[recent[-1],side,0]-uv[recent[-2],side,0])/max(1,recent[-1]-recent[-2])
                        predicted+=np.clip(velocity,-20,20)*min(3,index-recent[-1])
                    assignment_cost[ci,side]=np.median(np.linalg.norm(current-predicted,axis=1))
                else:
                    assignment_cost[ci,side]=abs(current[0,0]-(260 if side==0 else 850))
        ci_ids,side_ids=linear_sum_assignment(assignment_cost)
        for candidate_index,side in zip(ci_ids,side_ids):
            image_points,world_points,handedness=candidates[candidate_index]
            category=handedness[0]
''' +s[stop:]
    s=s.replace('left/right image ordering and recent wrist separation; hands do not cross, verified through all video frames','Hungarian assignment of full hand landmark sets to velocity-predicted anatomical tracks; initialized from source left/right positions; handover crossings supported')
    (root/'scripts/prepare_video.py').write_text(s)
    run([PY,str(root/'scripts/prepare_video.py')],root/'logs/prepare-video.log')
def templates(root,cfg):
    for name in ['calibrate_stereo.py','dense_stereo_depth.py','trtexec-python']:
        s=(PRIOR/'scripts'/name).read_text().replace('episode_000005',cfg['sequence'])
        s=re.sub(r'\b288\b',str(cfg['frames']),s)
        if name=='calibrate_stereo.py':
            from robust_stereo_template import adapt
            s=adapt(s)
        # Dense depth must include however many objects the actual video has.
        s=s.replace('in (1,2):',f"in range(1,{len(cfg['objects'])+1}):")
        if cfg.get('frame_cache_format')=='jpg':s=s.replace('frames/{i:06d}.png','frames/{i:06d}.jpg').replace('frames/000000.png','frames/000000.jpg')
        (root/'scripts'/name).write_text(s)
    # No duplicate long-lived weights or software environments.
def pot_spoon_reconstruction(root,cfg):
    assert cfg['objects']==['white_pot','wooden_spoon']
    rec=root/'reconstruction';assets=rec/'mesh_priors';assets.mkdir(exist_ok=True)
    # Keep the shared parent writable when a root-run GPU module emits files.
    (rec/'object_poses').mkdir(exist_ok=True)
    if not os.access(rec/'object_poses',os.W_OK):
        docker(['run','--rm','-v',f'{root}:/episode','--entrypoint','/bin/chown','flash_chord',f'{os.getuid()}:{os.getgid()}','/episode/reconstruction/object_poses'],root/'logs/fix-stage-directory-owner.log')
    for path in (PRIOR/'reconstruction/mesh_priors').glob('white_pot.*'):
        if not (assets/path.name).exists():shutil.copy2(path,assets/path.name)
    for name in ['reconstruct_spoon_asset.py','run_metric_stages.py','calibrate_table_camera.py','track_spoon_geometry.py','reconstruct_scene.py','refine_source_poses.py']:
        s=(PRIOR/'scripts'/name).read_text().replace('episode_000005',cfg['sequence']);s=re.sub(r'\b288\b',str(cfg['frames']),s)
        if name=='track_spoon_geometry.py':
            spans=cfg['support_intervals']['wooden_spoon'];s=s.replace('i<=43 or i>=231','any(a<=i<=b for a,b in '+repr(spans)+')')
        if name=='refine_source_poses.py':
            s=s.replace('(i<=74 or 112<=i<=143 or i>=231)',"any(a<=i<=z for a,z in "+repr(cfg['support_intervals']['white_pot'])+")")
            s=s.replace('(i<=43 or i>=231)',"any(a<=i<=z for a,z in "+repr(cfg['support_intervals']['wooden_spoon'])+")")
            s=s.replace("'white_pot':[[0,74],[112,143],[231,287]],'wooden_spoon':[[0,43],[231,287]]",repr(cfg['support_intervals'])[1:-1])
        if name=='reconstruct_scene.py':
            s=s.replace('for s in range(2):\n wrist=np.median(world_hands[:9,s,0],axis=0);world_hands[:9,s]+=wrist-world_hands[:9,s,0,None]\n','')
        if name=='run_metric_stages.py':
            s=s[:s.index('with concurrent.futures.ThreadPoolExecutor')]+'''for command,label,marker in [(slam,'slam','slam/poses/{last}.json'),(pose,'pot-poses','object_poses/white_pot/{last}.json')]:
    if not (ROOT/'reconstruction'/marker).exists():run(command,label)
print('Metric camera and pot pose estimation completed',flush=True)
'''.replace('{last}',f"{cfg['frames']-1:06d}")
        if name=='calibrate_table_camera.py':
            s=s.replace('if i in [0,45,90,135,180,225,270,287]:','if i in set(np.linspace(0,T-1,8).round().astype(int)):')
        (root/'scripts'/name).write_text(s)
    stages=[('reconstruct_spoon_asset','spoon-asset-reconstruction.json'),('run_metric_stages','object_poses/white_pot/'+f"{cfg['frames']-1:06d}.json"),('calibrate_table_camera','table-camera-calibration.json'),('track_spoon_geometry','spoon-geometric-tracking.json'),('reconstruct_scene','scene_motion.npz'),('refine_source_poses','scene_motion_physical.npz')]
    for name,marker in stages:
        if not (rec/marker).exists():run([PY,str(root/f'scripts/{name}.py')],root/f'logs/{name}.log')
    print(cfg['sequence'],'reconstruction complete',flush=True)
def metric(root,cfg):
    templates(root,cfg)
    for folder in ['depth','moge_intrinsics','masks']:(root/'reconstruction'/folder).mkdir(exist_ok=True)
    common=['run','--rm','--gpus','all','--ipc','host','-v',f'{root}:/episode','-v',f'{MODULES}:/workspace']
    jobs=[]
    for tag in ['sam2','moge']:
        command=json.loads((PRIOR/f'research/{tag}-command.json').read_text())[1:]
        command=[s.replace(str(PRIOR),str(root)) for s in command]
        completed=(root/'reconstruction/masks/1'/f"{cfg['frames']-1:06d}.png").exists() if tag=='sam2' else (root/'reconstruction/depth'/f"{cfg['frames']-1:06d}.png").exists()
        if tag=='moge':completed=completed or (root/'reconstruction/depth_moge'/f"{cfg['frames']-1:06d}.png").exists()
        if tag=='moge' and '--batch_size' in command:command[command.index('--batch_size')+1]='1'
        if not completed:jobs.append((command,root/f'logs/{tag}.log'))
    for command,log in jobs:docker(command,log)
    for name,marker in [('calibrate_stereo','camera-frame-calibration.json'),('dense_stereo_depth','stereo-depth-alignment.json')]:
        if not (root/'reconstruction'/marker).exists():run([PY,str(root/f'scripts/{name}.py')],root/f'logs/{name}.log')
    print(cfg['sequence'],'metric depth and segmentation complete',flush=True)
def generic_reconstruction(root,cfg):
    rec=root/'reconstruction';count=cfg['frames'];names=cfg['objects'];ext=cfg.get('frame_cache_format','png')
    (rec/'object_poses').mkdir(exist_ok=True)
    if cfg.get('own_geometry_required'):
        from own_geometry import build as build_own_geometry
        geometry=build_own_geometry(root,cfg)
    else:
        run([PY,str(BASE/'batch_scene.py'),'--run',str(root),'--stage','assets'],root/'logs/prepare-assets.log')
    if not (rec/'table-initial-plane.npz').exists():run([PY,str(BASE/'batch_scene.py'),'--run',str(root),'--stage','plane'],root/'logs/initial-table-plane.log')
    def write(name,s):
        s=s.replace('episode_000005',cfg['sequence']);s=re.sub(r'\b288\b',str(count),s)
        if ext=='jpg':s=s.replace('frames/{i:06d}.png','frames/{i:06d}.jpg').replace('frames/000000.png','frames/000000.jpg')
        (root/'scripts'/name).write_text(s)
    if not cfg.get('own_geometry_required') and 'wooden_spoon' in names and not (rec/'spoon-asset-reconstruction.json').exists():
        s=(PRIOR/'scripts/reconstruct_spoon_asset.py').read_text().replace('for b in [1,2]',f'for b in range(1,{len(names)+1})').replace('masks[1]',f"masks[{names.index('wooden_spoon')}]")
        write('reconstruct_spoon_asset.py',s);run([PY,str(root/'scripts/reconstruct_spoon_asset.py')],root/'logs/reconstruct_spoon_asset.log')
    if not (rec/f'slam/poses/{count-1:06d}.json').exists():
        shutil.copy2(BASE/'slam_empty_edges.py',root/'scripts/slam_empty_edges.py')
        args=json.loads((PRIOR/'research/slam-command.json').read_text())[1:];args=[s.replace(str(PRIOR),str(root)) for s in args]
        module_index=args.index('-m');assert args[module_index+1]=='v2d.droid_slam.lib.video_to_slam'
        args[module_index:module_index+2]=['/episode/scripts/slam_empty_edges.py']
        if cfg['episode']==37:args+=['--filter_thresh','0.15','--keyframe_thresh','0.5','--warmup','4']
        docker(args,root/'logs/slam.log')
    if not (rec/'table-camera-calibration.json').exists():
        s=(PRIOR/'scripts/calibrate_table_camera.py').read_text().replace('spoon-initial-geometry.npz','table-initial-plane.npz').replace('for b in [1,2]:',f'for b in range(1,{len(names)+1}):').replace('if i in [0,45,90,135,180,225,270,287]:','if i in set(np.linspace(0,T-1,8).round().astype(int)):')
        from robust_table_camera_template import adapt as table_adapt
        s=table_adapt(s)
        if cfg.get('camera_odometry_method')=='sift_pnp_metric':
            s=s.replace("REC/f'depth/{i:06d}.png'","REC/f'depth_foundation_stereo/{i:06d}.png'")
            s=s.replace("stereo=cv2.imread(str(REC/f'stereo_valid/{i:06d}.png'),0)>0","stereo=enc>0")
        write('calibrate_table_camera.py',s);run([PY,str(root/'scripts/calibrate_table_camera.py')],root/'logs/calibrate_table_camera.log')
    for b,name in enumerate(names):
        if (rec/f'object_poses/{name}/{count-1:06d}.json').exists():continue
        args=json.loads((PRIOR/'research/pot-poses-command.json').read_text())[1:]
        args=[s.replace(str(PRIOR),str(root)).replace('/masks/1',f'/masks/{b+1}').replace('white_pot',name) for s in args]
        if cfg.get('own_geometry_required'):
            args[args.index('--depth_folder')+1]='/episode/reconstruction/depth_foundation_stereo'
            args[args.index('--reference_frame')+1]=str(geometry['objects'][name]['reference_frame'])
        shutil.copy2(BASE/'batch_pose_memory.py',root/'scripts/pose_memory.py')
        index=args.index('-m');args[index:index+2]=['/episode/scripts/pose_memory.py']
        args+=['--reregister_iou_thresh','0.45']
        docker(args,root/f'logs/poses-{name}.log')
    print(cfg['sequence'],'camera and own reconstructed object poses ready' if cfg.get('own_geometry_required') else 'camera and public-asset poses ready',flush=True)
    if not (rec/'scene_motion_physical.npz').exists():run([PY,str(BASE/'batch_generic_finish.py'),'--run',str(root)],root/'logs/generic-scene-refinement.log')
if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--episode',type=int,required=True);p.add_argument('--stage',choices=['prepare','metric','pot_spoon','metric_pot_spoon','generic'],required=True);a=p.parse_args();root,cfg=context(a.episode)
    if a.stage=='prepare':prepare(root,cfg)
    if a.stage in ['metric','metric_pot_spoon']:metric(root,cfg)
    if a.stage in ['pot_spoon','metric_pot_spoon']:pot_spoon_reconstruction(root,cfg)
    if a.stage=='generic':generic_reconstruction(root,cfg)
