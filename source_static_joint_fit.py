"""Video-tested static hypotheses; joint fitting never uses pose ground truth."""
import cv2
import numpy as np
from scipy.optimize import minimize
from scipy.spatial.transform import Rotation
from sphere_marker_reference import marker_residual


def stable_windows(poses,eligible,minimum=8,maximum=40):
    windows=[];start=0
    while start<len(poses):
        if not eligible[start]:start+=1;continue
        end=start+1
        while end<len(poses) and eligible[end] and end-start<maximum:
            block=poses[start:end+1];center=np.median(block[:,:3,3],axis=0)
            rotation=Rotation.from_matrix(block[:,:3,:3]).mean().as_matrix()
            spread=np.max(np.linalg.norm(block[:,:3,3]-center,axis=1))
            angular=np.max(Rotation.from_matrix(rotation.T@block[:,:3,:3]).magnitude())
            if spread>.008 or angular>.10:break
            end+=1
        if end-start>=minimum:windows.append((start,end))
        start=end
    return windows


def refine(root,cfg,motion,renderers,photometric,records,hands,intr,mask_hands,visibility,measure,qualifies,templates):
    rec=root/'reconstruction';lookup={(row['frame'],row['object']):row for row in records};report=[]
    for body,name in enumerate(cfg['objects']):
        eligible=[]
        for frame in range(cfg['frames']):
            target=cv2.imread(str(rec/f'masks/{body+1}/{frame:06d}.png'),0)>0
            hand=cv2.dilate(mask_hands(hands,frame),np.ones((21,21),np.uint8))>0
            row=lookup[frame,name]
            # Hidden objects and any nearby visible hand are excluded. A
            # candidate still needs a rigid pose fitting every source frame.
            eligible.append(row['visibility']=='visible' and not (target&hand).any())
        for lo,hi in stable_windows(motion['object_to_world'][:,body],eligible):
            contexts=[];renderer=renderers[name]
            for frame in range(lo,hi):
                camera=np.linalg.inv(motion['camera_to_world'][frame])
                targets=[cv2.resize(cv2.imread(str(rec/f'masks/{i+1}/{frame:06d}.png'),0),(512,384),interpolation=cv2.INTER_NEAREST)>0 for i in range(len(cfg['objects']))]
                depths=[renderers[n].depth(camera@motion['object_to_world'][frame,i]) for i,n in enumerate(cfg['objects'])]
                valid=(cv2.resize(mask_hands(hands,frame),(512,384),interpolation=cv2.INTER_NEAREST)==0)&visibility(targets,depths,body)
                encoded=cv2.imread(str(rec/f'depth_stereo/{frame:06d}.png'),-1) if (rec/f'depth_stereo/{frame:06d}.png').exists() else None
                depth=cv2.resize(np.where(encoded>0,65535/np.maximum(encoded.astype(float),1)-1,0),(512,384),interpolation=cv2.INTER_NEAREST) if encoded is not None else None
                row=lookup[frame,name];spheres=row.get('sphere_observations',[])
                image=cv2.imread(str(rec/f"frames/{frame:06d}.{cfg.get('frame_cache_format','png')}"))
                rgb=cv2.resize(cv2.cvtColor(image,cv2.COLOR_BGR2RGB).astype(np.float32)/255.,(512,384))
                contexts.append((frame,camera,targets[body],valid,depth,rgb,spheres))
            def evaluate(world,context):
                frame,camera,target,valid,depth,rgb,spheres=context;pose=camera@world
                mask=renderer.render(pose);row=measure(mask,target,valid);surface=None;depth_error=None;photo_error=0.
                if depth is not None:
                    surface=renderer.depth(pose);keep=(surface>.05)&(depth>.05)&mask&target&valid
                    if keep.sum()>=20:depth_error=float(np.median(np.abs(surface[keep]-depth[keep])))
                photo=photometric.get(name)
                if photo is not None:
                    if surface is None:surface=renderer.depth(pose)
                    photo_error=photo.residual(pose,rgb,intr,surface,target&valid)[0]
                template=templates.get(name)
                if spheres and template:
                    xyz=[template['markers'][item['marker_id']]['center_m'] for item in spheres]
                    uv=[item['measurement']['xy'] for item in spheres]
                    row['sphere_error_px']=float(np.median(np.linalg.norm(marker_residual(pose,xyz,uv,intr),axis=1)))
                else:row['sphere_error_px']=None
                marker=row['sphere_error_px']
                loss=1-row['iou']+.03*photo_error+(.04*(np.sqrt(1+(depth_error/.01)**2)-1) if depth_error is not None else 0.)+(.06*(np.sqrt(1+(marker/3.)**2)-1) if marker is not None else 0.)
                return loss,row,depth_error,photo_error
            original=[evaluate(motion['object_to_world'][context[0],body],context) for context in contexts]
            block=motion['object_to_world'][lo:hi,body];seed=np.eye(4);seed[:3,3]=np.median(block[:,:3,3],axis=0);seed[:3,:3]=Rotation.from_matrix(block[:,:3,:3]).mean().as_matrix()
            sample=[contexts[i] for i in np.linspace(0,len(contexts)-1,min(8,len(contexts))).round().astype(int)]
            def pose(x):
                world=seed.copy();world[:3,3]+=x[:3];world[:3,:3]=Rotation.from_rotvec(x[3:]).as_matrix()@seed[:3,:3];return world
            result=minimize(lambda x:float(np.mean([evaluate(pose(x),c)[0] for c in sample])),np.zeros(6),method='Powell',bounds=[(-.008,.008)]*3+[(-.10,.10)]*3,options={'maxfev':160,'maxiter':6,'xtol':.0001,'ftol':.0001})
            candidate=pose(result.x);checked=[evaluate(candidate,c) for c in contexts]
            accepted=all(qualifies(r[1]) and (before[2] is None or r[2] is not None and r[2]<=before[2]+.002) and r[3]<=before[3]+.02 for before,r in zip(original,checked))
            report.append(dict(object=name,start_frame=lo,end_frame=hi-1,all_source_frames_checked=True,accepted=accepted,
                               mean_iou_before=float(np.mean([r[1]['iou'] for r in original])),mean_iou_after=float(np.mean([r[1]['iou'] for r in checked])),
                               maximum_position_change_m=float(np.max(np.linalg.norm(block[:,:3,3]-candidate[:3,3],axis=1)))))
            if accepted:
                motion['object_to_world'][lo:hi,body]=candidate
                for context,result in zip(contexts,checked):
                    row=lookup[context[0],name];row.update(result[1]);row.update(local_overlap_pass=qualifies(row),static_joint_fit=True)
                    if row['sphere_observations'] and templates.get(name):
                        xyz=[templates[name]['markers'][item['marker_id']]['center_m'] for item in row['sphere_observations']]
                        projected=marker_residual(context[1]@candidate,xyz,np.zeros((len(xyz),2)),intr)
                        for item,point in zip(row['sphere_observations'],projected):item['fitted_xy']=point.tolist()
    return dict(method='Joint rigid world pose over source-stable, untouched, visible segments; every frame rechecked for silhouette, metric depth and texture agreement',
                ground_truth_used=False,windows=report,accepted_windows=sum(row['accepted'] for row in report))
