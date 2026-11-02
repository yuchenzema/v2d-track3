"""Source-camera reconstruction diagnostic, explicitly not a policy video."""
import argparse,importlib.util,json,subprocess
from pathlib import Path
import cv2,numpy as np,trimesh
from improvement_common import BATCH
p=argparse.ArgumentParser();p.add_argument('--run',type=Path,required=True);a=p.parse_args();r=a.run
rec=r/'reconstruction';motion=np.load(rec/'scene_motion_physical.npz');fit=np.load(rec/'brush_coupled_motion.npz')
spec=importlib.util.spec_from_file_location('preview_geometry',r/'scripts/source_geometry.py');geo=importlib.util.module_from_spec(spec);spec.loader.exec_module(geo)
vertices=np.asarray(trimesh.load(rec/'mesh_priors/mini_sweeper.ply',process=False).vertices)[::5]
out=BATCH/'episode30_pickup_repair_20261026';temp=out/'coupled-reconstruction-raw.mp4';dest=out/'brush-reconstruction-source-review.mp4'
reader=cv2.VideoCapture(str(rec/'calibrated.mp4'));writer=cv2.VideoWriter(str(temp),cv2.VideoWriter_fourcc(*'mp4v'),20,(1024,384));assert writer.isOpened()
for i in range(263):
    ok,frame=reader.read();assert ok
    original=cv2.resize(frame,(512,384));overlay=original.copy();C=np.linalg.inv(motion['camera_to_world'][i]);O=fit['object_to_world'][i]
    mask=geo.silhouette(vertices,C@O)
    contours,_=cv2.findContours(mask.astype('uint8'),cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_SIMPLE)
    cv2.drawContours(overlay,contours,-1,(70,255,70),1)
    point=(C@O@fit['grasp_wrist_in_object'])[:3,3];uv=geo.project(point[None])[0]*.5
    if np.isfinite(uv).all():cv2.circle(overlay,tuple(np.rint(uv).astype(int)),3,(255,160,50),-1)
    for image,label in [(original,f'Human source | frame {i} | {i/20:.2f}s'),(overlay,'Reconstruction diagnostic | NOT a policy rollout')]:
        cv2.rectangle(image,(0,0),(512,23),(0,0,0),-1);cv2.putText(image,label,(6,16),cv2.FONT_HERSHEY_SIMPLEX,.42,(255,255,255),1)
    writer.write(np.concatenate([original,overlay],axis=1))
reader.release();writer.release()
encoded=dest.with_name(dest.stem+'.encoded.mp4')
subprocess.run(['ffmpeg','-nostdin','-v','error','-y','-i',str(temp),'-c:v','libx264','-crf','20','-pix_fmt','yuv420p','-movflags','+faststart',str(encoded)],check=True)
encoded.replace(dest);temp.unlink();print(dest)
