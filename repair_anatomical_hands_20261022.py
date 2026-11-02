"""Recover missed anatomical hands without assigning duplicate detections to the other arm.
Scoped to the four no-handover episodes reviewed from their original videos.
"""
import argparse,json,shutil
from pathlib import Path
import cv2,numpy as np,mediapipe as mp
from mediapipe.tasks.python import BaseOptions
from mediapipe.tasks.python.vision import HandLandmarker,HandLandmarkerOptions,RunningMode
from improvement_common import save,utc
p=argparse.ArgumentParser();p.add_argument('--run',type=Path,required=True);a=p.parse_args();r=a.run;c=json.loads((r/'episode-config.json').read_text());assert c['episode'] in [19,20,30,33];assert not (r/'coherent-input-checksums.json').exists();rec=r/'reconstruction';marker=r/'research/anatomical-hand-repair.json'
if marker.exists():raise SystemExit(0)
data=np.load(rec/'hand_measurements.npz');out={k:data[k].copy() for k in data.files};uv=out['uv'];xyz=out['relative_xyz'];obs=out['observed'];score=out['score'];T=len(obs);method=np.where(obs,1,0).astype(np.int8);rejected=[];recovered=[]
shutil.copy2(rec/'hand_measurements.npz',rec/'hand_measurements_before_anatomical_repair.npz')
images=[cv2.imread(str(rec/f'frames/{i:06d}.png')) for i in range(T)];assert all(im is not None for im in images)
for i in range(T):
 if obs[i,0] and (uv[i,0,0,0]>640 or (obs[i,1] and np.linalg.norm(uv[i,0,0]-uv[i,1,0])<75)):
  obs[i,0]=False;method[i,0]=0;rejected.append({'frame':i,'side':'left','reason':'right-region or duplicate-right-hand detection'})
 if obs[i,1] and uv[i,1,0,0]<350:
  obs[i,1]=False;method[i,1]=0;rejected.append({'frame':i,'side':'right','reason':'left-region detection in a source sequence without handover'})
options=HandLandmarkerOptions(base_options=BaseOptions(model_asset_path=str(r/'models/hand_landmarker.task')),running_mode=RunningMode.IMAGE,num_hands=1,min_hand_detection_confidence=.2,min_hand_presence_confidence=.2,min_tracking_confidence=.2)
with HandLandmarker.create_from_options(options) as detector:
 for i,image in enumerate(images):
  for side in range(2):
   if obs[i,side]:continue
   x0,x1=(0,640) if side==0 else (384,1024);y0,y1=280,768;crop=image[y0:y1,x0:x1];rgb=np.ascontiguousarray(cv2.cvtColor(crop,cv2.COLOR_BGR2RGB));result=detector.detect(mp.Image(image_format=mp.ImageFormat.SRGB,data=rgb))
   if not result.hand_landmarks:continue
   points=np.array([[p.x*(x1-x0)+x0,p.y*(y1-y0)+y0] for p in result.hand_landmarks[0]]);other=1-side
   if (side==0 and points[0,0]>640) or (side==1 and points[0,0]<350):continue
   if obs[i,other] and np.linalg.norm(points[0]-uv[i,other,0])<75:continue
   uv[i,side]=points;xyz[i,side]=[[p.x,p.y,p.z] for p in result.hand_world_landmarks[0]];obs[i,side]=True;score[i,side]=result.handedness[0][0].score;method[i,side]=2;recovered.append({'frame':i,'side':['left','right'][side],'method':'independent_anatomical_crop_mediapipe'})
# Short unobserved prefixes/suffixes are recovered from image motion, with forward/backward consistency checks.
gray=[cv2.cvtColor(im,cv2.COLOR_BGR2GRAY) for im in images]
for side in range(2):
 valid=np.flatnonzero(obs[:,side]);assert len(valid)>T*.35,('Insufficient anatomical hand evidence',side,len(valid))
 for anchor,direction in [(int(valid[0]),-1),(int(valid[-1]),1)]:
  points=uv[anchor,side].astype(np.float32).reshape(-1,1,2);last=anchor
  for i in range(anchor+direction,T if direction>0 else -1,direction):
   if abs(i-anchor)>25:break
   tracked,ok,_=cv2.calcOpticalFlowPyrLK(gray[last],gray[i],points,None,winSize=(31,31),maxLevel=4,criteria=(cv2.TERM_CRITERIA_EPS|cv2.TERM_CRITERIA_COUNT,50,.01));back,ok2,_=cv2.calcOpticalFlowPyrLK(gray[i],gray[last],tracked,None,winSize=(31,31),maxLevel=4,criteria=(cv2.TERM_CRITERIA_EPS|cv2.TERM_CRITERIA_COUNT,50,.01));good=ok.ravel().astype(bool)&ok2.ravel().astype(bool)&(np.linalg.norm(back[:,0]-points[:,0],axis=1)<3)
   if good.sum()<10:break
   delta=np.median(tracked[good,0]-points[good,0],axis=0);tracked[~good,0]=points[~good,0]+delta;nextpoints=tracked[:,0]
   if (side==0 and nextpoints[0,0]>640) or (side==1 and nextpoints[0,0]<350):break
   uv[i,side]=nextpoints;xyz[i,side]=xyz[anchor,side];obs[i,side]=True;score[i,side]=.25;method[i,side]=3;recovered.append({'frame':i,'side':['left','right'][side],'method':'bidirectionally_checked_image_flow','tracking_anchor':anchor,'valid_points':int(good.sum())});points=tracked;last=i
out['measurement_method']=method;out['observed']=obs;out['uv']=uv;out['relative_xyz']=xyz;out['score']=score;np.savez_compressed(rec/'hand_measurements.npz',**out)
# Recompute metric PnP/stereo hand placement while retaining the freshly fitted object trajectories and cameras.
original=np.load(rec/'scene_motion_physical.npz');physical={k:original[k].copy() for k in original.files};script=r/'scripts/reconstruct_scene.py'
import subprocess
subprocess.run([str(r/'.venv-video/bin/python'),str(script)],check=True);rebuilt=np.load(rec/'scene_motion.npz');assert np.allclose(rebuilt['camera_to_world'],physical['camera_to_world']);physical['hand_joint_positions']=rebuilt['hand_joint_positions'];physical['observed']=obs;np.savez_compressed(rec/'scene_motion_physical.npz',**physical)
save(marker,{'updated_utc':utc(),'rejected_duplicate_or_wrong_anatomical_detections':rejected,'recovered_source_measurements':recovered,'measurement_method_codes':{'1':'original_valid_mediapipe','2':'independent_anatomical_crop_mediapipe','3':'image_flow_from_valid_source_hand'},'observed_frames_after':obs.sum(0).tolist(),'first_valid_source_frame':[int(np.flatnonzero(obs[:,s])[0]) for s in range(2)],'object_poses_and_camera_preserved':True,'source_no_handover_annotation_used':True});print('Repaired anatomical source hands',c['episode'],obs.sum(0).tolist(),flush=True)
