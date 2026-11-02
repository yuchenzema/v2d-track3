"""Disambiguate pitcher poses using the visible wrist and bounded rigid motion."""
import importlib.util,json,os,shutil
from pathlib import Path
import cv2,numpy as np,trimesh,pyarrow as pa,pyarrow.parquet as pq
from scipy.optimize import minimize
from scipy.spatial.transform import Rotation,Slerp
from improvement_common import BASE,BATCH,save,utc
from episode_execution_guard import episode_slot
from source_mask_rasterizer import MeshSilhouette
from source_table_constraint import constrain_pose
from batch_native import _docker,docker

R=BASE/'runs/episode_000020_newton_human_match_20261023_g004';REC=R/'reconstruction'
MARKER=R/'research/temporal-pitcher-source-fit.json'

def main():
    if MARKER.exists():return
    assert not (R/'coherent-input-checksums.json').exists()
    with episode_slot(R,20):
        cfg=json.loads((R/'episode-config.json').read_text());backup=REC/'scene_motion_before_temporal_refit.npz'
        if not backup.exists():shutil.copy2(REC/'scene_motion_physical.npz',backup)
        loaded=np.load(backup);motion={k:loaded[k].copy() for k in loaded.files};hands=np.load(REC/'hand_measurements.npz')
        checkpoint=R/'research/temporal-refit-checkpoint.npz';records_path=R/'research/temporal-refit-records.json'
        records=json.loads(records_path.read_text()) if records_path.exists() else []
        if checkpoint.exists():motion['object_to_world']=np.load(checkpoint)['object_to_world']
        done={x['frame'] for x in records}
        k=json.loads((REC/'camera_intrinsics.json').read_text());mesh=trimesh.load(REC/'mesh_priors/water_pitcher.ply',process=False)
        vertices,inverse=np.unique(mesh.vertices,axis=0,return_inverse=True);faces=np.unique(np.sort(inverse[mesh.faces],axis=1),axis=0);renderer=MeshSilhouette(vertices,faces,k)
        spec=importlib.util.spec_from_file_location('source_geometry',R/'scripts/source_geometry.py');geo=importlib.util.module_from_spec(spec);spec.loader.exec_module(geo)
        # Recover a constant human wrist point from the visible late grasp.
        # This is independent of the robot's different wrist/contact geometry.
        A=[];b=[]
        for frame in range(120,156):
            if not hands['observed'][frame,1]:continue
            M=np.linalg.inv(motion['camera_to_world'][frame])@loaded['object_to_world'][frame,1];rot=M[:3,:3];t=M[:3,3];u,v=hands['uv'][frame,1,0]-[k['cx'],k['cy']]
            A.extend([k['fx']*rot[0]-u*rot[2],k['fy']*rot[1]-v*rot[2]]);b.extend([u*t[2]-k['fx']*t[0],v*t[2]-k['fy']*t[1]])
        wrist=np.linalg.lstsq(A,b,rcond=None)[0];assert len(A)>=20 and np.linalg.norm(wrist)<.25
        cal=json.loads((REC/'table-camera-calibration.json').read_text());table=(np.r_[cal['support_center_xy_m'],cal['support_height_gauge_m']],cal['support_radius_m'],cal['support_height_gauge_m']+.0005)
        hulls=np.load(R/'evaluation/object-collision-hulls.npz');collision=np.concatenate([hulls[x] for x in hulls.files if x.startswith('water_pitcher__')])
        previous=loaded['object_to_world'][17,1].copy();previous2=None
        for frame in range(18,159):
            if frame in done:previous2=previous;previous=motion['object_to_world'][frame,1].copy();continue
            C=np.linalg.inv(motion['camera_to_world'][frame]);valid=~geo.human_occlusion(hands,frame)
            valid&=cv2.resize(cv2.imread(str(REC/f'masks/1/{frame:06d}.png'),0),(512,384),interpolation=cv2.INTER_NEAREST)==0
            target=cv2.resize(cv2.imread(str(REC/f'masks/2/{frame:06d}.png'),0),(512,384),interpolation=cv2.INTER_NEAREST)>0;target&=valid
            observed=bool(hands['observed'][frame,1]);uv=hands['uv'][frame,1,0]*.5
            def pose(x):
                d=x[:3].copy();a=x[3:].copy()
                d*=min(1.,.06/max(np.linalg.norm(d),1e-12));a*=min(1.,.45/max(np.linalg.norm(a),1e-12))
                M=previous.copy();M[:3,3]+=d;M[:3,:3]=Rotation.from_rotvec(a).as_matrix()@previous[:3,:3]
                return constrain_pose(M,collision,*table)
            def metrics(M):
                render=renderer.render(C@M)&valid;iou=float((render&target).sum()/max(1,(render|target).sum()))
                p=(C@M)[:3,:3]@wrist+(C@M)[:3,3];project=p[:2]/max(p[2],.01)*[k['fx']/2,k['fy']/2]+[k['cx']/2,k['cy']/2]
                error=float(np.linalg.norm(project-uv)) if observed else 0.
                return iou,error
            def loss(x):
                M=pose(x);iou,error=metrics(M);penalty=.08*(max(0.,error-3.)/8.)**2
                penalty+=.002*float(np.sum((x[:3]/.04)**2)+np.sum((x[3:]/.3)**2))
                if previous2 is not None:
                    accel=M[:3,3]-2*previous[:3,3]+previous2[:3,3];penalty+=.003*float(np.sum((accel/.02)**2))
                return 1-iou+penalty
            seed=loaded['object_to_world'][frame,1];delta=np.r_[seed[:3,3]-previous[:3,3],Rotation.from_matrix(seed[:3,:3]@previous[:3,:3].T).as_rotvec()]
            starts=[np.clip(delta,[-.04]*3+[-.26]*3,[.04]*3+[.26]*3),np.zeros(6)];best=None
            for start in starts:
                fit=minimize(loss,start,method='Powell',bounds=[(-.045,.045)]*3+[(-.27,.27)]*3,options={'maxfev':650,'maxiter':18,'xtol':.00008,'ftol':.0001})
                if best is None or fit.fun<best.fun:best=fit
                iou,error=metrics(pose(best.x))
                if iou>.93 and (not observed or error<5.):break
            chosen=pose(best.x);dense=geo.overlap(geo.silhouette(mesh.vertices,C@chosen),target,valid)['iou']
            step=float(Rotation.from_matrix(previous[:3,:3].T@chosen[:3,:3]).magnitude());translation=float(np.linalg.norm(chosen[:3,3]-previous[:3,3]))
            assert step<=.451 and translation<=.085,(frame,step,translation)
            motion['object_to_world'][frame,1]=chosen;records.append({'frame':frame,'source_iou':dense,'wrist_observed':observed,'wrist_error_px':error if observed else None,'angular_step_rad':step,'translation_step_m':translation})
            np.savez_compressed(checkpoint,object_to_world=motion['object_to_world']);save(records_path,records)
            row={'episode':20,'pid':os.getpid(),'process_script':Path(__file__).name,'updated_utc':utc(),'status':'running','phase':f'Temporal pitcher reconstruction: {len(records)}/141 held frames','description':'Visible human wrist constraints and bounded per-frame rigid motion; no policy rollout yet','run':str(R),'activity_kind':'complete_reconstruction_and_retraining'}
            save(R/'research/rebuild-status.json',row);save(BATCH/'improvements_20261021/activities/episode_000020-temporal-pitcher-refit.json',row)
            print(frame,'IoU',round(dense,3),'wristpx',round(error,2) if observed else 'hidden','stepdeg',round(step*180/np.pi,2),flush=True)
            previous2=previous;previous=chosen
        end=motion['object_to_world'][158,1].copy();rest=motion['object_to_world'][162,1].copy()
        interpolate=Slerp([0.,1.],Rotation.from_matrix([end[:3,:3],rest[:3,:3]]))
        for frame in range(159,162):
            fraction=(frame-158)/4.;M=np.eye(4);M[:3,:3]=interpolate([fraction]).as_matrix()[0];M[:3,3]=(1-fraction)*end[:3,3]+fraction*rest[:3,3];motion['object_to_world'][frame,1]=constrain_pose(M,collision,*table)
        rotations=Rotation.from_matrix(motion['object_to_world'][:,1,:3,:3]);steps=(rotations[:-1].inv()*rotations[1:]).magnitude()
        assert steps.max()<.6,('Remaining orientation discontinuity',float(steps.max()))
        np.savez_compressed(REC/'scene_motion_physical.npz',**motion)
        for path in (R/'human_motion_data').glob('track3_*/processed/sequence_id=episode_000020/robot_name=sharpa_wave/motion.parquet'):
            row=pq.ParquetFile(path).read().to_pylist()[0];P=np.asarray(row['object_body_position']);Q=np.asarray(row['object_body_wxyz']);P[18:162,1]=motion['object_to_world'][18:162,1,:3,3];Q[18:162,1]=Rotation.from_matrix(motion['object_to_world'][18:162,1,:3,:3]).as_quat(scalar_first=True);row['object_body_position']=P.tolist();row['object_body_wxyz']=Q.tolist();pq.write_table(pa.Table.from_pylist([row]),path,compression='zstd')
        fits=json.loads((R/'research/collision-aware-grasps.json').read_text())['grasps']
        for fit in fits:
            existing=R/f"research/physical-hold-{fit['object']}-verified.json"
            if existing.exists():
                grip=json.loads(existing.read_text());fit.update(wrist_in_object=grip['wrist_in_object'],finger_joints=grip['finger_joints'])
        save(R/'research/temporal-source-grasps.json',{'grasps':fits})
        text=(BASE/'apply_collision_grasps.py').read_text().replace('human_motion_data/track3_coherent_variant/processed','human_motion_data/track3_collision_variant/processed').replace("R/'research/collision-aware-grasps.json'","R/'research/temporal-source-grasps.json'").replace("R/'research/collision-aware-applied-grasps.json'","R/'research/temporal-source-grasps-applied.json'").replace("shutil.copy(p.parent.parent.parent.parent/f'reconstructed_stage/{seq}_support.usda',stage);","None; # same reference support\n")
        script=R/'scripts/apply_temporal_source_grasps.py';script.write_text(text);retarget=cfg['native_container'].removesuffix('newton')+'retarget';_docker(['start',retarget]);docker(['exec',retarget,'/isaac-sim/python.sh','-u','/episode/scripts/apply_temporal_source_grasps.py'],R/'logs/temporal-source-grasps-applied.log')
        for name in ['physical-hold-water_pitcher-verified.json','physical-holds-verified.json']:
            p=R/'research'/name
            if p.exists():shutil.move(str(p),str(p.with_name('before-temporal-'+name)))
        p=R/'videos/reconstruction-source-review.mp4'
        if p.exists():shutil.move(str(p),str(R/'research/before-temporal-source-review.mp4'))
        save(MARKER,{'updated_utc':utc(),'method':__doc__,'human_wrist_in_object':wrist.tolist(),'maximum_rotation_step_rad':float(steps.max()),'frames':records,'exact_source_motion_verified':False,'physical_grasp_revalidation_required':True})

if __name__=='__main__':main()
