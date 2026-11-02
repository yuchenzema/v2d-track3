"""Discover bounded wrist feedback, then validate whole uniform native cohorts."""
import importlib.util,json,os,shutil,sys,traceback
import cv2,numpy as np,trimesh
from improvement_common import BASE,BATCH,save,utc,better
from batch_native import docker
from batch_review import review
from video_catalog import catalog

EP=int(sys.argv[1]) if len(sys.argv)>1 else 32
assert EP in [30,32]
R=BASE/'runs'/('episode_000032_newton_backward_grasp_20261020' if EP==32 else 'episode_000030_newton_consistent_brush_20261021')
CFG=json.loads((R/'episode-config.json').read_text());STEP=60000256 if EP==32 else 45001728
ACTIVITY=BATCH/f'improvements_20261021/activities/episode_{EP:06d}-robot-tracking.json'


def activity(status,phase,**extra):
    save(ACTIVITY,{'episode':EP,'pid':os.getpid(),'process_script':__file__.split('/')[-1],
        'updated_utc':utc(),'status':status,'phase':phase,
        'description':'Testing bounded robot wrist feedback on the trained '+('spoon/cup' if EP==32 else 'brush/rack')+' policy. Object forces and state remain unassisted.',**extra})


def main():
    activity('running','32 diagnostic controller settings; no new policy promoted')
    profiles=[{'left':left,'right':right,'maximum_correction_m':.015} for left in [0.,.25,.5,1.,2.] for right in [0.,.25,.5,1.,2.]]
    profiles +=[{'left':left,'right':right,'maximum_correction_m':.005} for left,right in [(0.,.5),(.5,0.),(.5,.5),(0.,1.),(1.,0.),(1.,1.),(2.,2.)]]
    if EP==30:profiles=[{'left':.5,'right':gain,'maximum_correction_m':cap,'right_wrist_residual_gain':wrist,'right_finger_residual_gain':finger}
        for wrist in [0.,.25] for finger in [.5,1.] for gain in [0.,.5,1.,2.] for cap in [.01,.03]]
    assert len(profiles)==32
    dst=R/'research/robot-tracking-grid-20261021';dst.mkdir(exist_ok=True)
    for name in ['record_native_cohort.py','evaluate_reference_controller.py','robot_object_tracking_feedback.py']:shutil.copy2(BASE/name,R/'scripts'/name)
    state=dst/'discovery-states.npz'
    if not state.exists():
        docker(['exec',CFG['native_container'],'env','V2D_RECORD_COHORT_SIZE=32','V2D_WRIST_VELOCITY_GAIN=0',
            f'V2D_REFERENCE_OVERRIDE=/episode/human_motion_data/{CFG["training_reference"]}/processed/sequence_id={CFG["sequence"]}/robot_name=sharpa_wave',
            'V2D_SIM_SUBSTEPS=20',f'V2D_SIM_NCONMAX={CFG.get("physics_nconmax",512)}',f'V2D_SIM_NJMAX={CFG.get("physics_njmax",1536)}','V2D_ROBOT_TRACKING_PROFILES='+json.dumps(profiles),
            'V2D_STATE_OUTPUT=/episode/'+str(state.relative_to(R)),'python','-u','/episode/scripts/record_native_cohort.py',
            f'evaluation.checkpoint=/episode/flash_chord_coherent_actor_transfer/policy_{STEP}.safetensors','evaluation.world_count=32','logging.mode=disabled',
            'evaluation.metrics_output=/episode/'+str((dst/'discovery-native-metrics.json').relative_to(R)),'hydra.run.dir=/episode/hydra/robot-tracking-discovery'],dst/'discovery.log')
    poses=np.load(state.with_name(state.stem+'-cohort-objects.npz'))['achieved'];assert poses.shape[1:3]==(32,2)
    cfg=json.loads((R/'episode-config.json').read_text());motion=np.load(R/'reconstruction/scene_motion_physical.npz');measure=np.load(R/'reconstruction/hand_measurements.npz')
    spec=importlib.util.spec_from_file_location('tracking_grid_geometry',R/'scripts/source_geometry.py');g=importlib.util.module_from_spec(spec);spec.loader.exec_module(g)
    vertices=[]
    for name in cfg['objects']:
        v=np.array(trimesh.load(R/f'reconstruction/mesh_priors/{name}.ply',process=False).vertices);vertices.append(v[np.linspace(0,len(v)-1,min(6000,len(v))).astype(int)])
    scores=np.zeros((32,cfg['frames'],2));finite=np.isfinite(poses).all(axis=(0,2,3))
    activity('running',f'Scoring all{CFG["frames"]} source frames for each diagnostic controller')
    for frame in range(cfg['frames']):
        C=np.linalg.inv(motion['camera_to_world'][frame]);valid=~g.human_occlusion(measure,frame)
        masks=[cv2.resize(cv2.imread(str(R/f'reconstruction/masks/{b+1}/{frame:06d}.png'),0),(512,384),interpolation=cv2.INTER_NEAREST)>0 for b in range(2)]
        native=round(frame*(len(poses)-1)/(cfg['frames']-1))
        for world in np.flatnonzero(finite):
            for b in range(2):scores[world,frame,b]=g.overlap(g.silhouette(vertices[b],C@g.matrix(poses[native,world,b])),masks[b],valid&~masks[1-b])['iou']
    ranked=sorted([{'world':w,'profile':p,'mean_iou':float(scores[w].mean()),'object_mean_iou':scores[w].mean(0).tolist()} for w,p in enumerate(profiles) if finite[w]],key=lambda x:x['mean_iou'],reverse=True)
    save(dst/'discovery-results.json',{'diagnostic_only':True,'source_frames_checked':cfg['frames'],'profiles':ranked,
        'world_grid_not_eligible_for_publication':True,'discovery_mesh_vertex_cap':6000,'uniform_validation_uses_complete_meshes':True})
    print('Top diagnostic feedback profiles',ranked[:3],flush=True)
    accepted=catalog(EP,BASE/f'runs/episode_{EP:06d}_newton_batch_20261015',True)['best']['selected']
    results=[]
    for candidate in ranked[:2]:
        activity('running','Independent uniform32-world validation; predetermined video world0',candidate=candidate)
        row=review(EP,STEP,root_override=R,container_override=CFG['native_container'],substeps_override=20,
            recording_world_count=32,tracking_feedback=candidate['profile'])
        eligible=better(row,accepted);row['passes_current_accepted_policy_gate']=eligible
        path=dst/f'uniform-world{candidate["world"]}-review.json';save(path,row);results.append({'review':str(path),'passes_current_accepted_policy_gate':eligible})
        print('Uniform feedback',candidate['profile'],'meanIoU',row['source_mean_object_iou'],'nativeSR',row['completion_rate_32_worlds'],'better',eligible,flush=True)
    save(dst/'completion.json',{'updated_utc':utc(),'status':'completed','uniform_reviews':results,'publication_pending':any(x['passes_current_accepted_policy_gate'] for x in results)})
    activity('completed','Uniform controller checks finished; videos unchanged pending publication review',results=results)


if __name__=='__main__':
    try:main()
    except Exception:
        activity('failed','Controller diagnostic failed; previous policy retained',error=traceback.format_exc());raise
