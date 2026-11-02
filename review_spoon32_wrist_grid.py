"""Discover small robot wrist corrections, then revalidate uniform full rollouts."""
import importlib.util,json,shutil
import cv2,numpy as np,trimesh
from improvement_common import BASE,BATCH,save,better
from batch_native import docker
from batch_review import review
from video_catalog import catalog

R=BASE/'runs/episode_000032_newton_backward_grasp_20261020'


def main():
    profiles=[{}]
    for side in ['left','right']:
        for axis in range(3):
            for value in [-.01,-.005,.005,.01]:
                v=[0.,0.,0.];v[axis]=value;profiles.append({side:v})
    for x,y in [(-.005,-.005),(-.005,.005),(.005,-.005),(.005,.005)]:profiles.append({'right':[x,y,0.]})
    for z in [-.005,.005]:profiles.append({'left':[0.,0.,z],'right':[0.,0.,z]})
    profiles.append({'right':[0.,0.,0.]});assert len(profiles)==32
    dst=R/'research/wrist-calibration-grid-20261021';dst.mkdir(exist_ok=True)
    for name in ['diagnostic_wrist_bias_grid.py','record_native_cohort.py','evaluate_reference_controller.py']:
        shutil.copy2(BASE/name,R/'scripts'/name)
    state=dst/'discovery-states.npz'
    if not state.exists():
        docker(['exec','v2d-backward-episode32-newton','env','V2D_RECORD_COHORT_SIZE=32','V2D_WRIST_VELOCITY_GAIN=0',
                'V2D_REFERENCE_OVERRIDE=/episode/human_motion_data/track3_refined_spoon_grasp_candidate/processed/sequence_id=episode_000032/robot_name=sharpa_wave',
                'V2D_SIM_SUBSTEPS=20','V2D_SIM_NCONMAX=512','V2D_SIM_NJMAX=1536','V2D_WRIST_BIAS_PROFILES='+json.dumps(profiles),
                'V2D_STATE_OUTPUT=/episode/'+str(state.relative_to(R)),'python','-u','/episode/scripts/record_native_cohort.py',
                'evaluation.checkpoint=/episode/flash_chord_coherent_actor_transfer/policy_60000256.safetensors','evaluation.world_count=32','logging.mode=disabled',
                'evaluation.metrics_output=/episode/'+str((dst/'native-metrics.json').relative_to(R)),'hydra.run.dir=/episode/hydra/wrist-calibration-grid'],dst/'discovery.log')
    poses=np.load(state.with_name(state.stem+'-cohort-objects.npz'))['achieved'];assert poses.shape[1:3]==(32,2)
    cfg=json.loads((R/'episode-config.json').read_text());motion=np.load(R/'reconstruction/scene_motion_physical.npz');measure=np.load(R/'reconstruction/hand_measurements.npz')
    spec=importlib.util.spec_from_file_location('wrist_grid_geometry',R/'scripts/source_geometry.py');g=importlib.util.module_from_spec(spec);spec.loader.exec_module(g)
    vertices=[]
    for n in cfg['objects']:
        v=np.array(trimesh.load(R/f'reconstruction/mesh_priors/{n}.ply',process=False).vertices);vertices.append(v[np.linspace(0,len(v)-1,min(6000,len(v))).astype(int)])
    T=cfg['frames'];scores=np.zeros((32,T,2));finite=np.isfinite(poses).all(axis=(0,2,3))
    for frame in range(T):
        C=np.linalg.inv(motion['camera_to_world'][frame]);valid=~g.human_occlusion(measure,frame)
        masks=[cv2.resize(cv2.imread(str(R/f'reconstruction/masks/{b+1}/{frame:06d}.png'),0),(512,384),interpolation=cv2.INTER_NEAREST)>0 for b in range(2)]
        native=round(frame*(len(poses)-1)/(T-1))
        for world in np.flatnonzero(finite):
            for b in range(2):scores[world,frame,b]=g.overlap(g.silhouette(vertices[b],C@g.matrix(poses[native,world,b])),masks[b],valid&~masks[1-b])['iou']
    ranked=sorted([{'world':w,'profile':p,'mean_iou':float(scores[w].mean()),'object_mean_iou':scores[w].mean(0).tolist()} for w,p in enumerate(profiles) if finite[w]],key=lambda x:x['mean_iou'],reverse=True)
    save(dst/'discovery-results.json',{'diagnostic_only':True,'source_frames_checked':T,'profiles':ranked,'discovery_worlds_not_eligible_for_publication':True,
         'discovery_mesh_vertex_cap':6000,'uniform_revalidation_uses_complete_meshes':True})
    print('Diagnostic top profiles',ranked[:3],flush=True)
    for candidate in ranked[:3]:
        bias=candidate['profile']
        if not bias:continue
        # Validate one hand correction at a time; combined-hand candidates remain diagnostic.
        if len(bias)!=1:continue
        side=next(iter(bias));grasp=next(v for v in cfg['grasps'] if v['side']==side)
        profile={'wrist_position':1.,'wrist_orientation':1.,'finger':1.,'after_source_frame':max(0,grasp['start']-4),'ramp_source_frames':8,
                 'until_source_frame':grasp['end'],'fade_source_frames':min(4,T-1-grasp['end']),'wrist_position_bias_m':bias}
        row=review(32,60000256,root_override=R,container_override='v2d-backward-episode32-newton',substeps_override=20,residual_profile=profile)
        path=dst/f'uniform-world{candidate["world"]}-review.json';save(path,row)
        best=catalog(32,BASE/'runs/episode_000032_newton_batch_20261015',True)['best']['selected'];qualifies=better(row,best)
        print('Uniform profile',bias,'IoU',row['source_mean_object_iou'],'SR',row['completion_rate_32_worlds'],'better',qualifies,flush=True)
        if qualifies:
            from publish_contact_controller_candidate import publish
            publish(32,path,f'spoon32-wrist-{candidate["world"]}',root_override=R)


if __name__=='__main__':main()
