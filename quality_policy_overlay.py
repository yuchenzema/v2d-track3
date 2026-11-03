"""Source-clock video overlays of genuine, unchanged official policy objects."""
import json,subprocess
from pathlib import Path
import cv2,numpy as np,trimesh,pyarrow.parquet as pq
from scipy.spatial.transform import Rotation
from improvement_common import save,digest,utc
from source_mask_rasterizer import MeshSilhouette


def render(root):
    from own_submission_queue import eligible
    root=Path(root);item=eligible(root);cfg=json.loads((root/'episode-config.json').read_text())
    source=Path(cfg['fast_source_run'])/'reconstruction';camera=np.load(source/'scene_motion_physical.npz')['camera_to_world']
    intr=json.loads((source/'camera_intrinsics.json').read_text());reference=np.load(root/'reconstruction/scene_motion_physical.npz')['object_to_world']
    table=pq.read_table(item['trajectory']).to_pandas();result=json.loads((root/'research/fast-submission-result.json').read_text())
    folder=root/f"evaluation/quality-visualization/{item['checkpoint_sha256'][:16]}";folder.mkdir(parents=True,exist_ok=True)
    movie=folder/'comparison.mp4';manifest=folder/'manifest.json'
    identity=dict(checkpoint_sha256=item['checkpoint_sha256'],trajectory_sha256=item['trajectory_sha256'],
        frozen_input_manifest_sha256=result['frozen_input_manifest_sha256'],camera_sha256=digest(source/'scene_motion_physical.npz'),
        mesh_sha256={name:digest(root/f'reconstruction/mesh_priors/{name}.ply') for name in cfg['objects']},
        source_frame_count=cfg['frames'],fps=cfg['fps'],overlay_code_sha256=digest(Path(__file__)))
    if manifest.exists() and movie.exists():
        old=json.loads(manifest.read_text())
        if old.get('identity')==identity and old.get('video_sha256')==digest(movie):return old
    renderers=[];actual=[]
    for slot,name in enumerate(cfg['objects']):
        mesh=trimesh.load(root/f'reconstruction/mesh_priors/{name}.ply',force='mesh',process=False)
        renderers.append(MeshSilhouette(mesh.vertices,mesh.faces,intr))
        rows=table[table.object_body_name==name].sort_values('step')
        assert len(rows)==cfg['frames'] and np.array_equal(rows.reference_frame.to_numpy(),np.arange(cfg['frames'])),'Overlay must preserve the full official source clock'
        poses=np.repeat(np.eye(4)[None],cfg['frames'],axis=0)
        poses[:,:3,3]=rows[[f'achieved_position_{axis}' for axis in 'xyz']].to_numpy()
        poses[:,:3,:3]=Rotation.from_quat(rows[[f'achieved_quaternion_{axis}' for axis in 'xyzw']].to_numpy()).as_matrix();actual.append(poses)
    temporary=folder/'comparison.tmp.mp4';palette=[(50,220,255),(255,100,200),(80,255,130)]
    command=['ffmpeg','-hide_banner','-loglevel','error','-y','-f','rawvideo','-pixel_format','bgr24','-video_size','1024x416',
        '-framerate',str(cfg['fps']),'-i','pipe:0','-an','-c:v','libx264','-threads','2','-preset','veryfast','-crf','23','-pix_fmt','yuv420p','-movflags','+faststart',str(temporary)]
    with (folder/'encode.log').open('w') as log:
        process=subprocess.Popen(command,stdin=subprocess.PIPE,stderr=log)
        try:
            for frame in range(cfg['frames']):
                image=cv2.imread(str(source/f'frames/{frame:06d}.png'));assert image is not None,f'Missing original frame {frame}'
                image=cv2.resize(image,(512,384));panels=[image.copy(),image.copy()];view=np.linalg.inv(camera[frame])
                for slot,renderer in enumerate(renderers):
                    color=np.array(palette[slot%len(palette)],dtype=np.uint8)
                    for panel,pose in zip(panels,[view@reference[frame,slot],view@actual[slot][frame]]):
                        mask=renderer.render(pose);panel[mask]=np.rint(.65*panel[mask]+.35*color).astype(np.uint8)
                        contours,_=cv2.findContours(mask.astype(np.uint8),cv2.RETR_LIST,cv2.CHAIN_APPROX_SIMPLE);cv2.drawContours(panel,contours,-1,tuple(map(int,color)),1)
                canvas=np.zeros((416,1024,3),np.uint8);canvas[32:,:512]=panels[0];canvas[32:,512:]=panels[1]
                cv2.putText(canvas,f'Source CAD estimate | frame {frame+1}/{cfg["frames"]}',(8,21),cv2.FONT_HERSHEY_SIMPLEX,.48,(255,255,255),1)
                cv2.putText(canvas,f'Policy objects | model {result["actual_training_steps"]:,.0f} steps',(520,21),cv2.FONT_HERSHEY_SIMPLEX,.48,(255,255,255),1)
                process.stdin.write(canvas.tobytes())
            process.stdin.close();code=process.wait();assert code==0,f'Overlay encoder failed: {folder}/encode.log'
        except BaseException:
            if process.poll() is None:process.kill();process.wait()
            raise
    probe=json.loads(subprocess.check_output(['ffprobe','-v','error','-count_frames','-show_streams','-of','json',str(temporary)],text=True))['streams'][0]
    assert int(probe['nb_read_frames'])==cfg['frames'],'Overlay dropped or duplicated source frames'
    temporary.replace(movie)
    record=dict(identity=identity,updated_utc=utc(),video=str(movie),video_sha256=digest(movie),actual_model_training_steps=result['actual_training_steps'],
        policy_inference=True,object_assistance=0.,world_count=1,source_clock_preserved=True,frame_splicing_used=False,
        reference_is_estimated=True,visualization_scope='Object silhouettes over original source RGB',strict_episode_completion=False,not_kaggle_score=True)
    save(manifest,record);return record
