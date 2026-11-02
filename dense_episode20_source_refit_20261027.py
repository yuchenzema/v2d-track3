"""Refit held pitcher poses with triangle silhouettes; preserve physical evidence."""
import copy,hashlib,importlib.util,json,os,shutil,subprocess
from pathlib import Path
import cv2,numpy as np,trimesh,pyarrow as pa,pyarrow.parquet as pq
from scipy.optimize import minimize
from scipy.spatial.transform import Rotation
from improvement_common import BASE,BATCH,save,utc
from episode_execution_guard import episode_slot
from source_mask_rasterizer import MeshSilhouette
from source_table_constraint import constrain_pose
from batch_native import _docker,docker

R=BASE/'runs/episode_000020_newton_human_match_20261023_g004'
REC=R/'reconstruction';MARKER=R/'research/triangle-pitcher-source-fit.json'

def main():
    if json.loads((BATCH/'user-pause.json').read_text()).get('status')=='paused_by_user':return
    if MARKER.exists():return
    assert not (R/'coherent-input-checksums.json').exists()
    with episode_slot(R,20):
        cfg=json.loads((R/'episode-config.json').read_text());backup=REC/'scene_motion_before_triangle_refit.npz'
        if not backup.exists():shutil.copy2(REC/'scene_motion_physical.npz',backup)
        loaded=np.load(backup);motion={k:loaded[k].copy() for k in loaded.files}
        checkpoint=R/'research/triangle-refit-checkpoint.npz';records_path=R/'research/triangle-refit-records.json'
        records=json.loads(records_path.read_text()) if records_path.exists() else []
        if checkpoint.exists():motion['object_to_world']=np.load(checkpoint)['object_to_world']
        done={item['frame'] for item in records}
        spec=importlib.util.spec_from_file_location('source_geometry',R/'scripts/source_geometry.py');geo=importlib.util.module_from_spec(spec);spec.loader.exec_module(geo)
        mesh=trimesh.load(REC/'mesh_priors/water_pitcher.ply',process=False)
        vertices,inverse=np.unique(mesh.vertices,axis=0,return_inverse=True);faces=np.unique(np.sort(inverse[mesh.faces],axis=1),axis=0)
        renderer=MeshSilhouette(vertices,faces,json.loads((REC/'camera_intrinsics.json').read_text()))
        hands=np.load(REC/'hand_measurements.npz');cal=json.loads((REC/'table-camera-calibration.json').read_text())
        table=(np.r_[cal['support_center_xy_m'],cal['support_height_gauge_m']],cal['support_radius_m'],cal['support_height_gauge_m']+.0005)
        hulls=np.load(R/'evaluation/object-collision-hulls.npz');collision=np.concatenate([hulls[k] for k in hulls.files if k.startswith('water_pitcher__')])
        previous=None
        for frame in range(18,159):
            if json.loads((BATCH/'user-pause.json').read_text()).get('status')=='paused_by_user':return
            if frame in done:
                saved=next(item for item in records if item['frame']==frame)
                previous=np.asarray(saved['parameters']) if saved['accepted'] else None
                continue
            camera=np.linalg.inv(motion['camera_to_world'][frame]);seed=loaded['object_to_world'][frame,1].copy()
            valid=~geo.human_occlusion(hands,frame)
            valid&=cv2.resize(cv2.imread(str(REC/f'masks/1/{frame:06d}.png'),0),(512,384),interpolation=cv2.INTER_NEAREST)==0
            target=cv2.resize(cv2.imread(str(REC/f'masks/2/{frame:06d}.png'),0),(512,384),interpolation=cv2.INTER_NEAREST)>0;target&=valid
            def pose(x):
                M=seed.copy();M[:3,3]+=x[:3];M[:3,:3]=Rotation.from_rotvec(x[3:]).as_matrix()@seed[:3,:3]
                return constrain_pose(M,collision,*table)
            def score(x):
                mask=renderer.render(camera@pose(x))&valid
                return float((mask&target).sum()/max(1,(mask|target).sum()))
            def loss(x):
                regularizer=.0005*float(np.sum((x[:3]/.12)**2)+np.sum((x[3:]/1.5)**2))
                continuity=0. if previous is None else .0002*float(np.sum(((x[:3]-previous[:3])/.03)**2)+np.sum(((x[3:]-previous[3:])/.35)**2))
                return 1-score(x)+regularizer+continuity
            original_score=geo.overlap(geo.silhouette(mesh.vertices,camera@seed),target,valid)['iou']
            zero=np.zeros(6);best=zero;best_loss=loss(zero)
            starts=[previous.copy(),zero] if previous is not None else [zero]
            for start in starts:
                fit=minimize(loss,np.clip(start,[-.15]*3+[-1.8]*3,[.15]*3+[1.8]*3),method='Powell',bounds=[(-.15,.15)]*3+[(-1.8,1.8)]*3,
                             options={'maxfev':800,'maxiter':24,'ftol':.00008,'xtol':.00008})
                if fit.fun<best_loss:best=fit.x.copy();best_loss=float(fit.fun)
                if score(best)>=.94:break
            candidate=pose(best);new_score=geo.overlap(geo.silhouette(mesh.vertices,camera@candidate),target,valid)['iou']
            accepted=bool(target.sum()>=60 and new_score>=original_score+.005)
            if accepted:motion['object_to_world'][frame,1]=candidate
            previous=best.copy() if accepted else None
            record={'frame':frame,'source_iou_before':original_score,'source_iou_after':new_score if accepted else original_score,
                    'accepted':accepted,'parameters':best.tolist() if accepted else [0.]*6,
                    'translation_change_m':float(np.linalg.norm(candidate[:3,3]-seed[:3,3])) if accepted else 0.,
                    'rotation_change_rad':float(Rotation.from_matrix(candidate[:3,:3]@seed[:3,:3].T).magnitude()) if accepted else 0.}
            records.append(record);np.savez_compressed(checkpoint,object_to_world=motion['object_to_world']);save(records_path,records)
            progress={'episode':20,'pid':os.getpid(),'process_script':Path(__file__).name,'updated_utc':utc(),'status':'running',
                'phase':f'Dense pitcher source-pose refinement: {len(records)}/141 held frames',
                'description':'Triangle silhouettes, unchanged metric geometry, finite-table constraints, independent dense-source validation',
                'run':str(R),'completed_cases':len(records),'total_cases':141,'activity_kind':'complete_reconstruction_and_retraining'}
            save(R/'research/rebuild-status.json',progress);save(BATCH/'improvements_20261021/activities/episode_000020-dense-pitcher-refit.json',progress)
            print(frame,round(original_score,3),'->',round(record['source_iou_after'],3),'accepted',accepted,flush=True)
        np.savez_compressed(REC/'scene_motion_physical.npz',**motion)
        for path in (R/'human_motion_data').glob('track3_*/processed/sequence_id=episode_000020/robot_name=sharpa_wave/motion.parquet'):
            row=pq.ParquetFile(path).read().to_pylist()[0];P=np.asarray(row['object_body_position']);Q=np.asarray(row['object_body_wxyz'])
            P[18:159,1]=motion['object_to_world'][18:159,1,:3,3];Q[18:159,1]=Rotation.from_matrix(motion['object_to_world'][18:159,1,:3,:3]).as_quat(scalar_first=True)
            row['object_body_position']=P.tolist();row['object_body_wxyz']=Q.tolist();pq.write_table(pa.Table.from_pylist([row]),path,compression='zstd')
        # Preserve the verified left grip; re-anchor the right-hand template to
        # the newly estimated pitcher poses before the physical hold search.
        fits=json.loads((R/'research/collision-aware-grasps.json').read_text())['grasps']
        left=json.loads((R/'research/physical-hold-blue_cup-verified.json').read_text())
        for fit in fits:
            if fit['object']=='blue_cup':fit.update(wrist_in_object=left['wrist_in_object'],finger_joints=left['finger_joints'])
        save(R/'research/triangle-source-grasps.json',{'grasps':fits})
        text=(BASE/'apply_collision_grasps.py').read_text().replace('human_motion_data/track3_coherent_variant/processed','human_motion_data/track3_collision_variant/processed')
        text=text.replace("R/'research/collision-aware-grasps.json'","R/'research/triangle-source-grasps.json'")
        text=text.replace("R/'research/collision-aware-applied-grasps.json'","R/'research/triangle-source-grasps-applied.json'")
        text=text.replace("shutil.copy(p.parent.parent.parent.parent/f'reconstructed_stage/{seq}_support.usda',stage);","None; # same reference support\n")
        script=R/'scripts/apply_triangle_source_grasps.py';compile(text,str(script),'exec');script.write_text(text)
        retarget=cfg['native_container'].removesuffix('newton')+'retarget';_docker(['start',retarget])
        docker(['exec',retarget,'/isaac-sim/python.sh','-u','/episode/scripts/apply_triangle_source_grasps.py'],R/'logs/triangle-source-grasps-applied.log')
        invalid=R/'research/before-triangle-pose-repair';invalid.mkdir(exist_ok=True)
        for path in [R/'research/physical-hold-water_pitcher-verified.json',R/'research/physical-holds-verified.json',R/'videos/reconstruction-source-review.mp4',*list((R/'evaluation').glob('physicalhold20-water-pitcher-*.json'))]:
            if path.exists():shutil.move(str(path),str(invalid/path.name))
        for filename in ['dense_episode20_source_refit_20261027.py','source_mask_rasterizer.py','source_mask_rasterizer.c','repair_episode20_physical_holds_20261027.py']:
            shutil.copy2(BASE/filename,R/'scripts'/filename)
        save(MARKER,{'updated_utc':utc(),'method':__doc__,'producer_source_sha256':{name:hashlib.sha256((BASE/name).read_bytes()).hexdigest() for name in ['dense_episode20_source_refit_20261027.py','source_mask_rasterizer.py','source_mask_rasterizer.c']},'source_frames_refitted':141,'source_geometry_rescaled':False,
            'source_time_warping_used':False,'policy_states_edited':False,'physical_grasp_revalidation_required':True,
            'mean_held_source_iou_before':float(np.mean([x['source_iou_before'] for x in records])),
            'mean_held_source_iou_after':float(np.mean([x['source_iou_after'] for x in records])),'frames':records})
        print('Triangle source refit complete; resume episode20 pipeline. Exact matching remains unverified.',flush=True)

if __name__=='__main__':main()
