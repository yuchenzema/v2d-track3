"""Reconstruct an occluded rigid tool from its observed human grasp and every visible silhouette."""
from pathlib import Path
import json,argparse,importlib.util
import numpy as np,cv2,trimesh
from scipy.spatial.transform import Rotation,Slerp
from scipy.signal import savgol_filter
parser=argparse.ArgumentParser();parser.add_argument('--run',type=Path,required=True);parser.add_argument('--object',default='wooden_spoon');parser.add_argument('--output-name',default='scene_motion_hand_guided_candidate.npz');parser.add_argument('--report-name',default='hand-guided-tool-reconstruction-candidate.json');a=parser.parse_args();r=a.run.resolve();rec=r/'reconstruction';cfg=json.loads((r/'episode-config.json').read_text());b=cfg['objects'].index(a.object);grasp=next(g for g in cfg['grasps'] if g['side']=='right' and g['object']==b);start,end=grasp['start'],grasp['end'];d=np.load(rec/'scene_motion_physical.npz');out={k:d[k].copy() for k in d.files};H=d['hand_joint_positions'][:,1];T=len(H);F=np.tile(np.eye(4),(T,1,1));y=H[:,9]-H[:,0];y/=np.linalg.norm(y,axis=1)[:,None];x=H[:,5]-H[:,17];x-=y*np.sum(x*y,axis=1)[:,None];x/=np.linalg.norm(x,axis=1)[:,None];z=np.cross(x,y);F[:,:3,:3]=np.stack([x,y,z],axis=-1);F[:,:3,3]=H[:,0];mesh=trimesh.load(rec/f'mesh_priors/{a.object}.ply',process=False);spec=importlib.util.spec_from_file_location('geometry',r/'scripts/source_geometry.py');g=importlib.util.module_from_spec(spec);spec.loader.exec_module(g);measure=np.load(rec/'hand_measurements.npz');before=json.loads((r/'evaluation/reference-source-fidelity.json').read_text());scores=np.array([f['objects'][a.object]['iou'] for f in before['frames']]);O=d['object_to_world'][:,b];local=np.linalg.inv(F)@O;indices=np.arange(start,end+1);good=indices[(scores[indices]>.75)&measure['observed'][indices,1]];assert len(good)>=6,len(good);q=Rotation.from_matrix(local[good,:3,:3]);distance=np.array([np.minimum((q.inv()*q[i]).magnitude(),1.)**2 for i in range(len(q))]);medoid=int(np.argmin((distance*scores[good][None]).sum(1)));angles=(q[medoid].inv()*q).magnitude();anchors=good[angles<.6];V=np.eye(4);V[:3,:3]=Rotation.from_matrix(local[anchors,:3,:3]).mean(weights=scores[anchors]**2).as_matrix();V[:3,3]=np.median(local[anchors,:3,3],axis=0);results=[];previous=np.zeros(6)
for i in range(start,end+1):
 C=np.linalg.inv(d['camera_to_world'][i]);target=cv2.resize(cv2.imread(str(rec/f'masks/{b+1}/{i:06d}.png'),0),(512,384),interpolation=cv2.INTER_NEAREST)>0;valid=~g.human_occlusion(measure,i)
 for other in range(len(cfg['objects'])):
  if other!=b:valid&=cv2.resize(cv2.imread(str(rec/f'masks/{other+1}/{i:06d}.png'),0),(512,384),interpolation=cv2.INTER_NEAREST)==0
 def evaluate(p):
  delta=np.eye(4);delta[:3,:3]=Rotation.from_rotvec(p[3:]).as_matrix();delta[:3,3]=p[:3];M=F[i]@delta@V;render=g.silhouette(mesh.vertices,C@M);score=g.overlap(render,target,valid)['iou'];penalty=.015*np.sum((p[:3]/.03)**2)+.01*np.sum((p[3:]/.3)**2)+.015*np.sum(((p[:3]-previous[:3])/.02)**2)+.008*np.sum(((p[3:]-previous[3:])/.15)**2);return score-penalty,M,score
 p=previous*.8;best=evaluate(p)
 for translation,rotation in [(.012,.08),(.006,.04),(.003,.02)]:
  for repeat in range(2):
   improved=False
   for dim in range(6):
    selected=None
    for sign in [-1,1]:
     trial=p.copy();trial[dim]+=sign*(translation if dim<3 else rotation)
     if np.linalg.norm(trial[:3])>.04 or np.linalg.norm(trial[3:])>.4:continue
     result=evaluate(trial)
     if result[0]>best[0]+1e-6:selected=trial;best=result
    if selected is not None:p=selected;improved=True
   if not improved:break
 previous=p.copy();out['object_to_world'][i,b]=best[1];results.append({'frame':i,'raw_iou':float(scores[i]),'guided_iou':best[2],'hand_frame_offset_m':float(np.linalg.norm(p[:3])),'hand_frame_offset_rad':float(np.linalg.norm(p[3:]))})
 if i%30==0:print(i,results[-1],flush=True)
# Blend only the pickup/release boundaries, retaining the original source grid.
for lo,hi in [(max(0,start-5),start),(end,min(T-1,end+5))]:
 for i in range(lo,hi+1):
  edge=start if i<=start else end;weight=(i-lo)/max(1,hi-lo) if edge==start else (hi-i)/max(1,hi-lo);prediction=F[i]@np.linalg.inv(F[edge])@out['object_to_world'][edge,b];M=O[i].copy();M[:3,3]=(1-weight)*M[:3,3]+weight*prediction[:3,3];M[:3,:3]=Slerp([0,1],Rotation.from_matrix([M[:3,:3],prediction[:3,:3]]))([weight])[0].as_matrix();out['object_to_world'][i,b]=M
center=mesh.vertices.mean(0);M=out['object_to_world'][:,b];centroids=np.einsum('tij,j->ti',M[:,:3,:3],center)+M[:,:3,3];centroids[start:end+1]=savgol_filter(centroids[start:end+1],5,2,axis=0);Q=Rotation.from_matrix(M[:,:3,:3]);Q=Rotation.from_matrix([Q[max(0,i-1):min(T,i+2)].mean().as_matrix() for i in range(T)]);M[:,:3,:3]=Q.as_matrix();M[:,:3,3]=centroids-np.einsum('tij,j->ti',M[:,:3,:3],center);out['object_to_world'][:,b]=M;np.savez_compressed(rec/a.output_name,**out);report={'method':__doc__,'held_interval':[start,end],'reliable_source_anchor_frames':anchors.tolist(),'anchor_count':len(anchors),'source_hand_pose':'Independent MediaPipe/PnP and metric stereo on observed skin; no task-object pose proxy','object_in_hand':V.tolist(),'before_mean_iou_held':float(scores[start:end+1].mean()),'candidate_mean_iou_held_before_final_smoothing':float(np.mean([x['guided_iou'] for x in results])),'maximum_translation_step_m':float(np.linalg.norm(np.diff(M[:,:3,3],axis=0),axis=1).max()),'maximum_orientation_step_rad':float((Q[:-1].inv()*Q[1:]).magnitude().max()),'frames':results};(rec/a.report_name).write_text(json.dumps(report,indent=2));print(json.dumps({k:v for k,v in report.items() if k not in ['frames','object_in_hand','reliable_source_anchor_frames']},indent=2))
