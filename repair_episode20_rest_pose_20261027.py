"""Fit resting pitcher pixels using an independently stable gravity orientation."""
import json,hashlib,re,os
from pathlib import Path
import cv2,numpy as np,trimesh,pyarrow.parquet as pq
from scipy.spatial.transform import Rotation
from scipy.optimize import differential_evolution
from improvement_common import BASE,save,utc
from source_table_constraint import constrain_pose

r=BASE/'runs/episode_000020_newton_human_match_20261023_g003'
prior=BASE/'runs/episode_000020_newton_full_rebuild_20261023'
assert not (r/'coherent-input-checksums.json').exists()
assert json.loads((prior/'research/rebuild-physical-preflight.json').read_text())['passed']
for name in ['water_pitcher.ply','water_pitcher.obj']:
    assert hashlib.sha256((r/'reconstruction/mesh_priors'/name).read_bytes()).digest()==hashlib.sha256((prior/'reconstruction/mesh_priors'/name).read_bytes()).digest()
old=pq.ParquetFile(next((prior/'human_motion_data/track3_collision_variant/processed').rglob('motion.parquet'))).read().to_pylist()[0]
stable_R=Rotation.from_quat(old['object_body_wxyz'][0][1],scalar_first=True).as_matrix()
up_local=stable_R.T@np.array([0.,0.,1.])
rec=r/'reconstruction';loaded=np.load(rec/'scene_motion_physical.npz');motion={k:loaded[k].copy() for k in loaded.files}
np.savez_compressed(rec/'scene_motion_before_stable_rest_fit.npz',**motion)
mesh=trimesh.load(rec/'mesh_priors/water_pitcher.ply',process=False);allvertices=np.asarray(mesh.vertices);vertices=allvertices[np.linspace(0,len(allvertices)-1,8000).astype(int)]
h=np.load(r/'evaluation/object-collision-hulls.npz');collision=np.concatenate([h[k] for k in h.files if k.startswith('water_pitcher__')])
cal=json.loads((rec/'table-camera-calibration.json').read_text());table=(np.r_[cal['support_center_xy_m'],cal['support_height_gauge_m']],cal['support_radius_m'],cal['support_height_gauge_m']+.0005)
k=json.loads((rec/'camera_intrinsics.json').read_text());reports=[]
def silhouette(M,C,verts):
    points=verts@M[:3,:3].T+M[:3,3];points=points@C[:3,:3].T+C[:3,3]
    uv=np.rint(points[:,:2]/np.maximum(points[:,2:],.01)*[k['fx']/2,k['fy']/2]+[k['cx']/2,k['cy']/2]).astype(int)
    valid=(points[:,2]>.05)&(uv[:,0]>=0)&(uv[:,0]<512)&(uv[:,1]>=0)&(uv[:,1]<384)
    mask=np.zeros((384,512),np.uint8);mask[uv[valid,1],uv[valid,0]]=1
    mask=cv2.morphologyEx(mask,cv2.MORPH_CLOSE,np.ones((3,3),np.uint8));contours,_=cv2.findContours(mask,cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_SIMPLE);cv2.drawContours(mask,contours,-1,1,-1)
    return mask.astype(bool)
def overlap(a,b):return float((a&b).sum()/max(1,(a|b).sum()))
for lo,hi in [(0,14),(162,201)]:
    indexes=[lo,(lo+hi)//2,hi];samples=[]
    for i in indexes:
        target=cv2.resize(cv2.imread(str(rec/f'masks/2/{i:06d}.png'),0),(512,384),interpolation=cv2.INTER_NEAREST)>0
        samples.append((np.linalg.inv(motion['camera_to_world'][i]),target))
    seed=motion['object_to_world'][lo:hi+1,1].copy();P=np.median(seed[:,:3,3],axis=0)
    representative=seed[len(seed)//2,:3,:3];worldup=representative@up_local
    alignment=Rotation.align_vectors(np.array([[0.,0.,1.]]),worldup[None])[0].as_matrix()
    baseR=alignment@representative
    def pose(x):
        M=np.eye(4);M[:3,:3]=Rotation.from_euler('z',x[2]).as_matrix()@baseR;M[:3,3]=P+np.r_[x[:2],0.]
        return constrain_pose(M,collision,*table,settled=True)
    def loss(x):return -np.mean([overlap(silhouette(pose(x),C,vertices),target) for C,target in samples])+.002*float(np.sum((x[:2]/.03)**2))
    fit=differential_evolution(loss,[(-.045,.045),(-.045,.045),(-.4,.4)],seed=20261027,popsize=8,maxiter=24,polish=False)
    M=pose(fit.x);before=[];after=[]
    for i in range(lo,hi+1):
        C=np.linalg.inv(motion['camera_to_world'][i]);target=cv2.resize(cv2.imread(str(rec/f'masks/2/{i:06d}.png'),0),(512,384),interpolation=cv2.INTER_NEAREST)>0
        before.append(overlap(silhouette(motion['object_to_world'][i,1],C,allvertices),target));after.append(overlap(silhouette(M,C,allvertices),target));motion['object_to_world'][i,1]=M
    assert np.mean(after)>=np.mean(before)-.02,(lo,float(np.mean(before)),float(np.mean(after)))
    reports.append({'source_interval':[lo,hi],'source_mean_iou_before':float(np.mean(before)),'source_mean_iou_after':float(np.mean(after)),'fit_parameters':fit.x.tolist(),'stable_local_gravity_axis':up_local.tolist()})
    print(reports[-1],flush=True)
np.savez_compressed(rec/'scene_motion_physical.npz',**motion)
save(r/'research/stable-rest-source-fit.json',{'updated_utc':utc(),'method':__doc__,'reference':str(prior),'reports':reports,'physical_stability_revalidation_required':True,'policy_states_or_live_forces_edited':False})
