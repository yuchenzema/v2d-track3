"""Measure locating-sphere image references on the original source frame grid."""
import argparse
import hashlib
import json
from pathlib import Path
import subprocess
import cv2
import numpy as np
from improvement_common import BASE,BATCH,save,utc,digest
from sphere_marker_reference import measure_markers


def main():
    parser=argparse.ArgumentParser();parser.add_argument('--run',type=Path,required=True);args=parser.parse_args()
    root=args.run;cfg=json.loads((root/'episode-config.json').read_text());rec=root/'reconstruction'
    templates_path=BATCH/'focused_episodes_20261028/huggingface_audit_20261028/marker-templates.json'
    templates=json.loads(templates_path.read_text())['objects'];motion=np.load(rec/'scene_motion_physical.npz')
    intr=json.loads((rec/'camera_intrinsics.json').read_text());hands=np.load(rec/'hand_measurements.npz')
    source=BASE/f"datasets/video_to_data_challenge/track_3/evaluation/videos/chunk-000/observation.images.ego_cam_a/episode_{cfg['episode']:06d}.mp4"
    probe=cv2.VideoCapture(str(source));fps=probe.get(cv2.CAP_PROP_FPS);count=int(probe.get(cv2.CAP_PROP_FRAME_COUNT));probe.release()
    assert count==cfg['frames'] and count==len(motion['camera_to_world'])
    assert intr['width']==1024 and intr['height']==768
    signature=hashlib.sha256((digest(templates_path)+digest(source)+digest(rec/'camera_intrinsics.json')+hashlib.sha256(motion['camera_to_world'].tobytes()).hexdigest()).encode()).hexdigest()
    marker=root/'research/locating-sphere-tracking.json';tracks_path=rec/'locating-sphere-observations.json'
    if marker.exists() and tracks_path.exists() and json.loads(marker.read_text()).get('input_signature')==signature:return
    summary={name:{'template_markers':len(templates.get(name,{}).get('markers',[])),'observations':0,'frames_with_measurements':0,
                   'frames_with_three_or_more_measurements':0,'observations_per_marker':{},'initial_reprojection_errors_px':[]} for name in cfg['objects']}
    frames=[];temporary=root/'research/sphere-reference-diagnostic.tmp.mp4'
    writer=cv2.VideoWriter(str(temporary),cv2.VideoWriter_fourcc(*'mp4v'),fps,(1024,768));assert writer.isOpened()
    for frame in range(count):
        image=cv2.imread(str(rec/f"frames/{frame:06d}.{cfg.get('frame_cache_format','png')}"));assert image is not None
        canvas=image.copy();camera=np.linalg.inv(motion['camera_to_world'][frame]);occlusion=np.zeros(image.shape[:2],np.uint8)
        for side in range(2):
            if hands['observed'][frame,side]:
                points=np.rint(hands['uv'][frame,side]).astype(np.int32)
                cv2.fillConvexPoly(occlusion,cv2.convexHull(points[[0,1,5,9,13,17]]),1)
        row={'frame':frame,'time_s':frame/fps,'objects':{}}
        for body,name in enumerate(cfg['objects']):
            template=templates.get(name)
            if not template:
                row['objects'][name]={'template_available':False,'markers':[]};continue
            mask=cv2.imread(str(rec/f'masks/{body+1}/{frame:06d}.png'),0);assert mask is not None
            assert mask.shape==image.shape[:2]
            observed=measure_markers(image,template,camera@motion['object_to_world'][frame,body],intr,mask>0,occlusion)
            count_observed=sum(x['observed'] for x in observed);stats=summary[name]
            stats['observations']+=count_observed;stats['frames_with_measurements']+=int(count_observed>0);stats['frames_with_three_or_more_measurements']+=int(count_observed>=3)
            for item in observed:
                prediction=np.rint(item['predicted_xy']).astype(int)
                if np.all(np.abs(prediction)<10000):cv2.drawMarker(canvas,tuple(prediction),(0,90,255),cv2.MARKER_CROSS,8,1)
                if item['observed']:
                    stats['observations_per_marker'][str(item['marker_id'])]=stats['observations_per_marker'].get(str(item['marker_id']),0)+1
                    point=np.array(item['measurement']['xy']);stats['initial_reprojection_errors_px'].append(float(np.linalg.norm(point-item['predicted_xy'])))
                    cv2.circle(canvas,tuple(np.rint(point).astype(int)),5,(0,255,80),1)
            row['objects'][name]={'template_available':True,'markers':observed}
        cv2.rectangle(canvas,(0,0),(1024,52),(10,10,10),-1)
        cv2.putText(canvas,f'SOURCE SPHERE DIAGNOSTIC | frame {frame}/{count} | red: model prior, green: image measurement',(10,22),cv2.FONT_HERSHEY_SIMPLEX,.48,(255,255,255),1)
        cv2.putText(canvas,'Not a policy rollout. Unseen markers have no measurement.',(10,43),cv2.FONT_HERSHEY_SIMPLEX,.45,(255,255,255),1)
        writer.write(canvas);frames.append(row)
        if frame%40==0:
            save(root/'research/sphere-tracking-progress.json',{'updated_utc':utc(),'frames_checked':frame+1,'total_frames':count,'episode':cfg['episode']})
            print('measured locating spheres',frame+1,'/',count,flush=True)
    writer.release();video=root/'research/sphere-reference-diagnostic.mp4'
    subprocess.run(['ffmpeg','-hide_banner','-loglevel','error','-y','-i',str(temporary),'-c:v','libx264','-threads','2','-preset','fast','-crf','23','-pix_fmt','yuv420p','-movflags','+faststart',str(video)],check=True)
    temporary.unlink()
    for stats in summary.values():
        errors=stats.pop('initial_reprojection_errors_px')
        stats['median_initial_reprojection_error_px']=float(np.median(errors)) if errors else None
    save(tracks_path,{'episode':cfg['episode'],'source_frames':count,'source_fps':fps,'source_video_sha256':digest(source),'input_signature':signature,'frames':frames})
    result={'updated_utc':utc(),'episode':cfg['episode'],'input_signature':signature,'source_frames':count,'source_fps':fps,
            'objects':summary,'template_file':str(templates_path),'observations_file':str(tracks_path),
            'all_source_frames_checked':True,'occluded_markers_filled':False,'policy_states_edited':False,
            'official_marker_coordinates_available':False,'coordinates_derived_from_public_metric_scans':True,
            'camera_and_source_clock_preserved':True,'source_only_diagnostic':True,
            'video':'/artifacts/'+str(video.relative_to(BASE/'runs')),
            'limitation':'Assignments are constrained by scan geometry and the initial pose, and require pose/silhouette validation. Fewer than three visible spheres do not independently determine a full rigid pose.'}
    save(marker,result);print(json.dumps(result,indent=2),flush=True)


if __name__=='__main__':main()
