"""Prepare source image/mask inputs; never replace a released reconstruction."""
import json
from pathlib import Path
import urllib.request
import cv2
import numpy as np
from improvement_common import BASE,BATCH,save,utc,digest

OUT=BATCH/'sam3d_rebuild_20261026'
EXPECTED=[1,3,5,6,8,10,14,15,16,19,20,25,28,30,32,33,35,36,37,38]

def prepare():
    with urllib.request.urlopen('http://127.0.0.1:8768/api/status',timeout=30) as response:rows=json.load(response)['episodes']
    assert [r['episode'] for r in rows]==EXPECTED
    manifest={'prepared_utc':utc(),'method':'SAM 3D Objects','episodes':{},'jobs':[],
              'input_masks':'Existing SAM2 masks; not SAM3D-generated geometry',
              'training_started':False,'source_motion_verified':False}
    for row in rows:
        ep=row['episode'];run=Path(row['videos']['latest']['run'])
        if ep==1:run=BASE/'runs/episode_000001_newton_v5_20261014'
        rec=run/'reconstruction';video=BASE/f'datasets/video_to_data_challenge/track_3/evaluation/videos/chunk-000/observation.images.ego_cam_a/episode_{ep:06d}.mp4'
        masks=[]
        for body,name in enumerate(row['objects'],1):
            paths=sorted((rec/f'masks/{body}').glob('*.png'));assert len(paths)==row['source_frames'],(ep,name,len(paths))
            scored=[]
            for path in paths:
                mask=cv2.imread(str(path),cv2.IMREAD_GRAYSCALE);assert mask is not None
                y,x=np.nonzero(mask);area=len(x)
                edge=bool(area and (x.min()==0 or y.min()==0 or x.max()==mask.shape[1]-1 or y.max()==mask.shape[0]-1))
                scored.append((int(path.stem),area*(.25 if edge else 1),path,mask.shape))
            # One visible input from each third; fixed source timestamps retained.
            for partition in np.array_split(np.arange(len(scored)),3):
                frame,score,path,shape=max((scored[int(i)] for i in partition),key=lambda entry:entry[1]);assert score>30,(ep,name,'No visible mask')
                target=OUT/f'inputs/episode_{ep:06d}/{name}/frame_{int(frame):06d}'
                target.mkdir(parents=True,exist_ok=True)
                mask=cv2.imread(str(path),cv2.IMREAD_GRAYSCALE)
                assert cv2.imwrite(str(target/'mask.png'),mask)
                calibrated=next((rec/f'frames/{int(frame):06d}.{ext}' for ext in ['png','jpg'] if (rec/f'frames/{int(frame):06d}.{ext}').exists()),None)
                if calibrated is not None:
                    image=cv2.imread(str(calibrated));calibrated_source=str(calibrated);source_hash=digest(calibrated)
                else:
                    calibrated=rec/'calibrated.mp4';assert calibrated.exists(),(ep,frame,'Missing calibrated image/video')
                    clip=cv2.VideoCapture(str(calibrated));assert int(clip.get(cv2.CAP_PROP_FRAME_COUNT))==row['source_frames']
                    clip.set(cv2.CAP_PROP_POS_FRAMES,int(frame));ok,image=clip.read();clip.release();assert ok
                    calibrated_source=str(calibrated)+f'#frame={int(frame)}';source_hash=digest(calibrated)
                assert image is not None and image.shape[:2]==mask.shape,(ep,frame,'Frame/mask coordinate mismatch')
                assert cv2.imwrite(str(target/'image.png'),image)
                entry={'episode':ep,'object':name,'frame':int(frame),'source_time_s':int(frame)/20.,'image':str(target/'image.png'),
                       'mask':str(target/'mask.png'),'original_mask':str(path),'mask_sha256':digest(target/'mask.png'),'original_mask_sha256':digest(path),
                       'mesh':str(target/'mesh.glb'),'transform':str(target/'transform.json'),'intrinsics':str(target/'intrinsics.json'),
                       'source_video':str(video),'source_run':str(run),'mask_resolution':list(map(int,shape[::-1])),
                       'calibrated_source_frame':calibrated_source,'calibrated_source_file_sha256':source_hash,
                       'image_sha256':digest(target/'image.png'),
                       'image_coordinates':'Exact calibrated source frame matching the SAM2 mask; not a resized raw frame',
                       'source_camera_intrinsics':str(rec/'camera_intrinsics.json'),
                       'seed':20261026+ep+int(frame)}
                depth=rec/f'depth/{int(frame):06d}.png'
                if depth.exists():
                    encoded=cv2.imread(str(depth),cv2.IMREAD_UNCHANGED)
                    assert encoded is not None and encoded.shape==mask.shape and encoded.dtype==np.uint16,(ep,frame,'Depth coordinate/encoding mismatch')
                    entry.update(depth=str(depth),depth_sha256=digest(depth),depth_source='Existing stereo/MoGE metric estimate, not evaluation ground truth')
                masks.append(entry)
        capture=cv2.VideoCapture(str(video));frame=0
        while True:
            ok,image=capture.read()
            if not ok:break
            frame+=1
        capture.release();assert frame==row['source_frames'],(ep,frame)
        assert all(Path(j['image']).exists() for j in masks)
        manifest['episodes'][str(ep)]={'objects':row['objects'],'source_frames_decoded':frame,'source_video_sha256':digest(video),
                                     'status':'inputs_prepared','geometry_reconstructed':False,'policy_trained':False}
        manifest['jobs']+=masks
        print('Prepared SAM3D inputs:',ep,len(masks),'decoded frames:',frame,flush=True)
    save(OUT/'input-manifest.json',manifest)
    return manifest

if __name__=='__main__':prepare()
