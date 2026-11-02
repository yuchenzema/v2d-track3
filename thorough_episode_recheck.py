"""Fresh all-frame review of the exact releases currently shown in the GUI.
No simulation states, policies or existing releases are modified.
"""
import os,json,time,traceback,urllib.request,hashlib,subprocess,xml.etree.ElementTree as ET
from pathlib import Path
import cv2,numpy as np,trimesh
from scipy.spatial.transform import Rotation
from improvement_common import BASE,BATCH,save,utc,digest
from episode_clock import control_frames
OUT=BATCH/'human_video_recheck_20261021';EXPECTED=[1,3,5,6,8,10,14,15,16,19,20,25,28,30,32,33,35,36,37,38]
COLORS=[(70,240,90),(230,90,240),(250,190,60)]
def matrix(pose):
 m=np.eye(4);m[:3,3]=pose[:3];m[:3,:3]=Rotation.from_quat(pose[3:]).as_matrix();return m
def project(p,k):return p[:,:2]/np.maximum(p[:,2:],1e-9)*[k['fx'],k['fy']]+[k['cx'],k['cy']]
def silhouette(vertices,m,k):
 p=vertices@m[:3,:3].T+m[:3,3];u=np.rint(project(p,k)*.5).astype(np.int32);valid=(p[:,2]>.05)&(u[:,0]>=0)&(u[:,0]<512)&(u[:,1]>=0)&(u[:,1]<384);mask=np.zeros((384,512),np.uint8);mask[u[valid,1],u[valid,0]]=255;mask=cv2.morphologyEx(mask,cv2.MORPH_CLOSE,np.ones((3,3),np.uint8));contours,_=cv2.findContours(mask,cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_SIMPLE);cv2.drawContours(mask,contours,-1,255,-1);return mask>0
def occlusion(measure,i):
 mask=np.zeros((768,1024),np.uint8)
 for side in range(2):
  if not measure['observed'][i,side]:continue
  p=np.rint(measure['uv'][i,side]).astype(np.int32);cv2.fillConvexPoly(mask,cv2.convexHull(p[[0,1,5,9,13,17]]),255)
  for finger in range(5):
   ids=[0]+list(range(1+4*finger,5+4*finger))
   for a,b in zip(ids[:-1],ids[1:]):cv2.line(mask,tuple(p[a]),tuple(p[b]),255,14)
 return cv2.resize(mask,(512,384),interpolation=cv2.INTER_NEAREST)>0

def metric(a,b,valid):
 a=a&valid;b=b&valid;ya,xa=np.where(a);yb,xb=np.where(b)
 return {'iou':float((a&b).sum()/max(1,(a|b).sum())),'centroid_error_px':float(np.linalg.norm([xa.mean()-xb.mean(),ya.mean()-yb.mean()])*2) if len(xa) and len(xb) else None,'visible_source_pixels':int(b.sum()),'projected_pixels':int(a.sum())}
def intervals(bad):
 out=[];start=None
 for i,x in enumerate([*bad,False]):
  if x and start is None:start=i
  if not x and start is not None:out.append({'start_frame':start,'end_frame':i-1,'start_s':start/20,'end_s':(i-1)/20});start=None
 return out

def mean(values):return float(np.mean(values)) if len(values) else None

def artifact(path):return '/artifacts/'+str(Path(path).relative_to(BASE/'runs'))
def report_status(completed,ep=None,frame=0,total=0,status='running',**extra):
 save(OUT/'status.json',{'updated_utc':utc(),'pid':os.getpid(),'process_script':Path(__file__).name,'status':status,'episodes_completed':len(completed),'total_episodes':20,'current_episode':ep,'current_frame':frame,'current_total_frames':total,'source_frames_completed':sum(x['source_frames'] for x in completed),'episodes':completed,**extra})
 if ep is not None:
  save(BATCH/f'improvements_20261021/activities/episode_{ep:06d}-thorough-recheck.json',{'episode':ep,'pid':os.getpid(),'process_script':Path(__file__).name,'updated_utc':utc(),'status':status,'phase':f'Fresh human-video review: {frame} / {total} frames','description':'Rechecking every original frame against the currently linked rollout, object masks, reconstructed object poses and observed hand landmarks.','completed_cases':frame,'total_cases':total})

def recheck(row,completed):
 ep=row['episode'];release=row['videos']['latest'];root=Path(release['run']);cfg=json.loads((root/'episode-config.json').read_text()) if (root/'episode-config.json').exists() else {};recroot=BASE/'runs/episode_000001_newton_v5_20261014' if ep==1 and not cfg.get('validated_video_pipeline') else root;rec=recroot/'reconstruction';dst=OUT/f'episode_{ep:06d}';dst.mkdir(parents=True,exist_ok=True)
 T=row['source_frames'];names=row['objects'];motion=np.load(rec/'scene_motion_physical.npz');hands=np.load(rec/'hand_measurements.npz');k=json.loads((rec/'camera_intrinsics.json').read_text());statepath=Path(release['recorded_states']);state=np.load(statepath);poses=state['body_poses'];body_names=state['body_names'].tolist();N=len(poses)
 assert N==control_frames(dict(cfg,frames=T)) and np.isfinite(poses).all() and np.all(state['object_assistance']==0) and np.array_equal(state['reference_frames'],np.arange(N))
 from rollout_collision_validation import audit_rollout
 collision_validation=audit_rollout(root,statepath,release['version'],release.get('selected',{}).get('inference_reference'))
 meshes=[trimesh.load(rec/f'mesh_priors/{n}.ply',process=False).vertices for n in names];ids=[body_names.index(f'track3_{n}/object') for n in names]
 source=BASE/f'datasets/video_to_data_challenge/track_3/evaluation/videos/chunk-000/observation.images.ego_cam_a/episode_{ep:06d}.mp4';raw=cv2.VideoCapture(str(source));cal=cv2.VideoCapture(str(rec/'calibrated.mp4'));assert raw.isOpened() and cal.isOpened()
 rolloutpath=BASE/'runs'/release['links']['rollout'].split('/artifacts/')[1].split('?')[0];roll=cv2.VideoCapture(str(rolloutpath));assert roll.isOpened()
 cfg=json.loads((root/'episode-config.json').read_text()) if (root/'episode-config.json').exists() else {};review=json.loads((root/'research/source-video-review.json').read_text());grasps=cfg.get('grasps',review.get('grasps',[]))
 # Episode 1/5 source event annotations are stored in their native reference metadata.
 if not grasps:
  constraints=rec/'robot-grasp-constraints.json'
  if constraints.exists():
   data=json.loads(constraints.read_text())
   save(dst/'legacy-grasp-metadata.json',data)
 phases=review.get('phases',review.get('task_phases',[]));save(dst/'source-phase-annotations.json',{'phases':phases,'grasps':grasps,'automatic_uniform_windows_also_checked':True})
 tip_offsets={}
 for side in ['left','right']:
  xml=ET.parse(BASE/f'video_to_data/robotic_grounding/source/robotic_grounding/robotic_grounding/assets/xmls/sharpawave/{side}_sharpawave.xml')
  for finger in ['thumb','index','middle','ring','pinky']:tip_offsets[side,finger]=np.fromstring(xml.find(f'.//site[@name="{side}_{finger}_tip_site"]').get('pos'),sep=' ')
 output=dst/'source-policy-overlay.tmp.mp4';writer=cv2.VideoWriter(str(output),cv2.VideoWriter_fourcc(*'mp4v'),20,(1536,384));assert writer.isOpened()
 records=[];thumbnails=[];object_previous=None;object_reconstruction_previous=None
 for i in range(T):
  if i%20==0:report_status(completed,ep,i,T)
  ok,original=raw.read();ok2,image=cal.read();ok3,robot=roll.read();assert ok and ok2 and ok3,(ep,i,'source or rollout truncated');assert image.shape[:2]==(768,1024)
  source_state=int(np.rint(i*(N-1)/(T-1)));body=poses[source_state];C=np.linalg.inv(motion['camera_to_world'][i]);exclude=occlusion(hands,i);masks=[]
  for b in range(len(names)):
   mask=cv2.imread(str(rec/f'masks/{b+1}/{i:06d}.png'),0);assert mask is not None,(ep,i,b);masks.append(cv2.resize(mask,(512,384),interpolation=cv2.INTER_NEAREST)>0)
  overlay=cv2.resize(image,(512,384));frame={'frame':i,'time_s':i/20,'recorded_control_frame':source_state,'objects':{},'hands':{}}
  current=[];expected=[]
  for b,name in enumerate(names):
   M=matrix(body[ids[b]]);E=motion['object_to_world'][i,b];a=silhouette(meshes[b],C@M,k);e=silhouette(meshes[b],C@E,k);valid=~exclude
   for other in range(len(names)):
    if other!=b:valid&=~masks[other]
   score=metric(a,masks[b],valid);reference=metric(e,masks[b],valid);allscore=metric(a,masks[b],np.ones_like(valid));angle=(Rotation.from_matrix(E[:3,:3]).inv()*Rotation.from_matrix(M[:3,:3])).magnitude();position=float(np.linalg.norm(M[:3,3]-E[:3,3]))
   frame['objects'][name]={'rollout':score,'reconstruction':reference,'rollout_unoccluded':allscore,'source_mask_pixels':int(masks[b].sum()),'visible_source_fraction':float((masks[b]&valid).sum()/max(1,masks[b].sum())),'object_position_error_to_video_reconstruction_m':position,'object_orientation_error_to_video_reconstruction_rad':float(angle),'rollout_position_world_m':M[:3,3].tolist(),'reconstruction_position_world_m':E[:3,3].tolist()}
   if object_previous is not None:frame['objects'][name].update(rollout_step_m=float(np.linalg.norm(M[:3,3]-object_previous[b])),reconstruction_step_m=float(np.linalg.norm(E[:3,3]-object_reconstruction_previous[b])))
   frame['objects'][name]['rollout_orientation_world_xyzw']=Rotation.from_matrix(M[:3,:3]).as_quat().tolist()
   frame['objects'][name]['reconstruction_orientation_world_xyzw']=Rotation.from_matrix(E[:3,:3]).as_quat().tolist()
   current.append(M[:3,3]);expected.append(E[:3,3]);contours,_=cv2.findContours((a*255).astype(np.uint8),cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_SIMPLE);cv2.drawContours(overlay,contours,-1,COLORS[b%len(COLORS)],1)
  object_previous=np.array(current);object_reconstruction_previous=np.array(expected)
  for s,side in enumerate(['left','right']):
   points=[body[body_names.index(f'{side}_sharpa_wave/{side}_hand_C_MC'),:3]]
   for finger in ['thumb','index','middle','ring','pinky']:
    for segment in (['CMC_VL','MCP_VL','DP'] if finger=='thumb' else ['MCP_VL','MP','DP']):points.append(body[body_names.index(f'{side}_sharpa_wave/{side}_{finger}_{segment}'),:3])
    distal=body[body_names.index(f'{side}_sharpa_wave/{side}_{finger}_DP')];points.append(Rotation.from_quat(distal[3:]).apply(tip_offsets[side,finger])+distal[:3])
   points=np.array(points);uv=project(points@C[:3,:3].T+C[:3,3],k);valid=bool(hands['observed'][i,s]);errors=np.linalg.norm(uv-hands['uv'][i,s],axis=1)
   frame['hands'][side]={'observed':valid,'wrist_error_px':float(errors[0]) if valid else None,'mean_joint_error_px':float(errors.mean()) if valid else None,'fingertip_error_px':float(errors[[4,8,12,16,20]].mean()) if valid else None}
  panels=[cv2.resize(original,(512,384)),cv2.resize(robot,(512,384)),overlay]
  for panel,title in zip(panels,['Original human','Current policy video','Actual object poses over source']):
   cv2.rectangle(panel,(0,0),(512,25),(15,15,15),-1);cv2.putText(panel,f'{title} | frame {i} | {i/20:.2f}s',(7,18),cv2.FONT_HERSHEY_SIMPLEX,.44,(255,255,255),1,cv2.LINE_AA)
  combined=np.concatenate(panels,axis=1);writer.write(combined);records.append(frame)
  if i in set(np.linspace(0,T-1,12).round().astype(int)):thumbnails.append(combined)
 writer.release();assert not raw.read()[0] and not roll.read()[0],(ep,'unexpected extra source or rollout frames');raw.release();cal.release();roll.release()
 encoded=dst/f'source-policy-overlay.encoded-{os.getpid()}.mp4'
 subprocess.run(['ffmpeg','-hide_banner','-loglevel','error','-y','-i',str(output),'-c:v','libx264','-threads','2','-preset','fast','-crf','22','-pix_fmt','yuv420p','-movflags','+faststart',str(encoded)],check=True)
 encoded.replace(dst/'source-policy-overlay.mp4');output.unlink()
 for start in range(0,len(thumbnails),4):cv2.imwrite(str(dst/f'timeline-{start//4:02d}.jpg'),np.concatenate(thumbnails[start:start+4],axis=0))
 summaries={};worst=[]
 for name in names:
  values=[x['objects'][name] for x in records];scores=[v['rollout']['iou'] for v in values];ref=[v['reconstruction']['iou'] for v in values];bad=[s<.5 for s in scores];visible_bad=[s<.5 and v['rollout']['visible_source_pixels']>=30 for s,v in zip(scores,values)]
  held={i for g in grasps if names[g['object']]==name for i in range(g['start'],g['end']+1)};boundary={i for g in grasps if names[g['object']]==name for f in [g['start'],g['end']] for i in range(max(0,f-4),min(T,f+5))}
  windows=[]
  for start in range(0,T,20):
   end=min(T,start+20);windows.append({'start_frame':start,'end_frame':end-1,'mean_iou':mean(scores[start:end]),'reconstruction_mean_iou':mean(ref[start:end])})
  summaries[name]={'mean_iou':mean(scores),'reconstruction_mean_iou':mean(ref),'held_mean_iou':mean([scores[i] for i in sorted(held)]),'transition_mean_iou':mean([scores[i] for i in sorted(boundary)]),'source_frame_count':T,'source_frames_below_0_5_iou':sum(bad),'visible_frames_below_0_5_iou':sum(visible_bad),'first_low_overlap_source_frame':next((i for i,v in enumerate(bad) if v),None),'bad_intervals':intervals(bad),'worst_one_second_windows':sorted(windows,key=lambda x:x['mean_iou'])[:3],'native_tracking':{'mean_position_error_m':mean([v['object_position_error_to_video_reconstruction_m'] for v in values]),'max_position_error_m':max(v['object_position_error_to_video_reconstruction_m'] for v in values),'mean_orientation_error_rad':mean([v['object_orientation_error_to_video_reconstruction_rad'] for v in values])},'mean_centroid_error_px':mean([v['rollout']['centroid_error_px'] for v in values if v['rollout']['centroid_error_px'] is not None]),'low_source_visibility_frames':sum(v['rollout']['visible_source_pixels']<30 for v in values)}
  worst.extend((score,i,name) for i,score in enumerate(scores) if values[i]['rollout']['visible_source_pixels']>=30)
 # Additional evidence focuses on the worst visible failures, rather than just evenly spaced frames.
 worst.sort();selected=[]
 for score,i,name in worst:
  if all(abs(i-j)>8 for j in selected):selected.append(i)
  if len(selected)>=8:break
 cap=cv2.VideoCapture(str(dst/'source-policy-overlay.mp4'));sheets=[]
 for i in selected:
  cap.set(cv2.CAP_PROP_POS_FRAMES,i);ok,im=cap.read();assert ok;sheets.append(im)
 cap.release()
 for j in range(0,len(sheets),4):cv2.imwrite(str(dst/f'worst-{j//4:02d}.jpg'),np.concatenate(sheets[j:j+4],axis=0))
 discrepancies=sorted({i for name in names for i,v in enumerate(records) if v['objects'][name]['rollout']['iou']<.5});hand_summary={side:{'observed_frames':sum(f['hands'][side]['observed'] for f in records),'mean_wrist_error_px':mean([f['hands'][side]['wrist_error_px'] for f in records if f['hands'][side]['observed']]),'mean_joint_error_px':mean([f['hands'][side]['mean_joint_error_px'] for f in records if f['hands'][side]['observed']])} for side in ['left','right']}
 summary={'episode':ep,'version':release['version'],'updated_utc':utc(),'source_frames':T,'all_source_frames_decoded_and_scored':True,'native_control_frames':N,'source_video_sha256':digest(source),'recorded_states_sha256':digest(statepath),'run':str(root),'reconstruction_used':str(recroot),'objects':summaries,'hands':hand_summary,'mean_object_iou':mean([v['mean_iou'] for v in summaries.values()]),'frames_with_any_object_below_0_5_iou':len(discrepancies),'exact_human_match_verified':False,'needs_improvement':bool(discrepancies),'source_time_warping_used':False,'new_policy_training_performed':False,'method':'Fresh actual-state projection versus every source object mask, plus reconstruction/source comparison, observed hand projection, source/rollout full decoding and fixed native source-clock alignment. Vertex-silhouette approximation at512x384; human occlusion excluded explicitly; raw overlap also retained.','limitations':['Source masks, camera reconstruction and human landmark estimates have uncertainty.','Reconstruction alignment is measured separately; a good reconstructed-reference score cannot establish a true human match.','This is an offline recheck of the exact published attempt, not a new independent physical rollout.'],'links':{'report':artifact(dst/'report.json'),'frames':artifact(dst/'frame-metrics.json'),'video':artifact(dst/'source-policy-overlay.mp4'),'timeline':artifact(dst/'timeline-00.jpg'),'worst':artifact(dst/'worst-00.jpg')}}
 summary['collision_validation']=collision_validation
 save(dst/'frame-metrics.json',{'episode':ep,'version':release['version'],'frames':records});save(dst/'report.json',summary);report_status(completed,ep,T,T,status='completed');print(ep,'all',T,'frames checked; overlap',summary['mean_object_iou'],'discrepancy frames',len(discrepancies),flush=True);return summary

def main():
 OUT.mkdir(parents=True,exist_ok=True);snapshot=json.load(urllib.request.urlopen('http://127.0.0.1:8768/api/status'));assert [r['episode'] for r in snapshot['episodes']]==EXPECTED;save(OUT/'gui-release-snapshot.json',snapshot)
 with urllib.request.urlopen('https://huggingface.co/api/datasets/nvidia/video_to_data_challenge/tree/main/track_3/evaluation?recursive=true&limit=1000',timeout=45) as response:inventory=json.load(response)
 save(OUT/'huggingface-evaluation-inventory.json',inventory);source_checks=[]
 for entry in inventory:
  if entry['type']!='file' or not entry['path'].endswith('.mp4'):continue
  p=BASE/'datasets/video_to_data_challenge'/entry['path'];sha=digest(p);assert sha==entry['lfs']['oid'];source_checks.append({'path':entry['path'],'sha256':sha})
 assert len(source_checks)==60;save(OUT/'source-hash-verification.json',{'updated_utc':utc(),'all_60_videos_match_huggingface':True,'videos':source_checks})
 completed=[];report_status(completed)
 for row in snapshot['episodes']:
  result=recheck(row,completed);completed.append(result);report_status(completed)
 current=json.load(urllib.request.urlopen('http://127.0.0.1:8768/api/status'));changed=[r['episode'] for r in current['episodes'] if r['video_version']!=next(x['version'] for x in completed if x['episode']==r['episode'])]
 save(OUT/'summary.json',{'updated_utc':utc(),'expected_episodes':EXPECTED,'episodes_checked':20,'source_frames_checked':sum(r['source_frames'] for r in completed),'all_current_videos_rechecked':not changed,'changed_video_versions_since_start':changed,'all_60_source_video_hashes_verified':True,'episodes':completed,'exact_human_match_verified':False,'new_policy_training_performed':False});report_status(completed,status='completed');print('All20 published attempts rechecked, source frames',sum(r['source_frames'] for r in completed),flush=True)
if __name__=='__main__':
 try:main()
 except Exception:
  OUT.mkdir(parents=True,exist_ok=True);save(OUT/'failure.json',{'updated_utc':utc(),'error':traceback.format_exc()});raise
