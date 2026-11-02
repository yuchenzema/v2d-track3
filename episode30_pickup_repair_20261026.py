"""Isolated physical pickup repair; never accept hand-only motion as success."""
import json,os,shlex,shutil,subprocess,traceback
from pathlib import Path
from improvement_common import BASE,BATCH,save,utc,digest
from batch_native import _docker,docker

SOURCE=BASE/'runs/episode_000030_newton_full_rebuild_20261023'
ROOT=BASE/'runs/episode_000030_newton_pickup_repair_20261026'
OUT=BATCH/'episode30_pickup_repair_20261026'
NAME='v2d-pickup26-episode30-newton'
ACTIVITY=BATCH/'improvements_20261021/activities/episode_000030-physical-pickup-repair.json'

def status(state,phase,**extra):
    row={'episode':30,'pid':os.getpid(),'process_script':Path(__file__).name,'status':state,'phase':phase,
         'description':phase,'run':str(ROOT),'updated_utc':utc(),'activity_kind':'physical_pickup_repair',
         'sam3d_checkpoint_access':'still_required','policy_improvement_verified':False,**extra}
    save(OUT/'status.json',row);save(ACTIVITY,row)

def prepare():
    ROOT.mkdir(exist_ok=True);OUT.mkdir(exist_ok=True)
    if not (ROOT/'research/pickup-isolation.json').exists():
        for name in ['scripts','human_motion_data']:
            shutil.copytree(SOURCE/name,ROOT/name,dirs_exist_ok=True)
        for name in ['logs','research','evaluation','videos','deliverables','flash_chord_coherent_actor_transfer']:(ROOT/name).mkdir(exist_ok=True)
        rec=ROOT/'reconstruction';rec.mkdir(exist_ok=True)
        for p in (SOURCE/'reconstruction').iterdir():
            target=rec/p.name
            if p.is_file():shutil.copy2(p,target)
            elif p.name in ['mesh_priors','public_render_assets']:shutil.copytree(p,target,dirs_exist_ok=True)
            elif not target.exists():target.symlink_to(p,target_is_directory=True)
        for n in ['.venv-video','models']:
            if not (ROOT/n).exists():(ROOT/n).symlink_to((SOURCE/n).resolve(),target_is_directory=True)
        for n in ['source-video-review.json','source-review-verified.json','collision-aware-grasps.json']:
            if (SOURCE/'research'/n).exists():shutil.copy2(SOURCE/'research'/n,ROOT/'research'/n)
        cfg=json.loads((SOURCE/'episode-config.json').read_text())
        cfg.update(run=str(ROOT),native_container=NAME,training_variant='pickup_repair_20261026',
                   match_output=str(OUT),audit_roster=[30],training_initial_counter=0,training_budget=100000256,
                   reference_review_pending=True,requires_verified_improvement_for_publication=True,
                   improvement_experiment='Brush pickup repair with unassisted physical grasp and cold-start gates',
                   reconstruction_reused_from=str(SOURCE),sam3d_reconstruction_used=False)
        save(ROOT/'episode-config.json',cfg)
        fits=json.loads((SOURCE/'research/collision-aware-grasps.json').read_text())['grasps']
        brush=next(g for g in fits if g['object']=='mini_sweeper')
        save(ROOT/'evaluation/grasp-pose-search.json',{'best_candidates':{'mini_sweeper':{**brush,'world_inward_direction':[0,0,0],'inward_m':0.}}})
        prior=BASE/'runs/episode_000030_newton_observed_brush_20261021/evaluation/refined-grasp-search-source-34-handle.json'
        shutil.copy2(prior,ROOT/'research/previous-handle-grasp-seed.json')
        save(ROOT/'research/pickup-isolation.json',{'created_utc':utc(),'source':str(SOURCE),
             'original_geometry_copied_to_isolated_run':True,'original_reference_copied':True,'old_policy_or_optimizer_copied':False,
             'sam3d_reconstruction_used':False,'reason':'Diagnose physical pickup while SAM3D checkpoint access is blocked',
             'source_manifest_sha256':digest(SOURCE/'deliverables/manifest.json')})
    # The search writes only robot postures and its own diagnostic outputs.
    shutil.copy2(BASE/'diagnose_refined_grasp_search.py',ROOT/'scripts/diagnose_refined_grasp_search.py')
    shutil.copy2(BASE/'grasp_search_protocol.py',ROOT/'scripts/grasp_search_protocol.py')
    text=(BASE/'diagnose_static_grasp.py').read_text().replace("'sim.njmax=1536'","'sim.nconmax=2048','sim.njmax=4096'")
    (ROOT/'scripts/diagnose_static_grasp.py').write_text(text)
    if _docker(['ps','-a','--filter',f'name=^{NAME}$','--format','{{.Names}}']).stdout.strip()!=NAME:
        info=json.loads(_docker(['inspect','v2d-rebuild23-episode30-newton']).stdout)[0]
        cmd=['run','-d','--gpus','all','--runtime','nvidia','--ipc','host','--name',NAME]
        for mount in info['Mounts']:
            cmd+=['-v',(str(ROOT) if mount['Destination']=='/episode' else mount['Source'])+':'+mount['Destination']+('' if mount['RW'] else ':ro')]
        cmd+=['-e','XLA_PYTHON_CLIENT_PREALLOCATE=false','-e','PYGLET_HEADLESS=true','--workdir','/workspace','--entrypoint','/bin/sleep',info['Image'],'infinity']
        _docker(cmd,ROOT/'logs/create-pickup-container.log')
    else:_docker(['start',NAME])

def anatomical_retry():
    status('running','Rebuilding pickup from the video-tracked right hand and pixel-checked brush orientation branches')
    subprocess.run([str(ROOT/'.venv-video/bin/python'),str(BASE/'rebuild_episode30_anatomical_pickup.py'),'--run',str(ROOT)],check=True)
    reference='track3_pickup_anatomical';label='anatomical-source34-grasp'
    if not (ROOT/f'evaluation/{label}.json').exists():
        docker(['exec',NAME,'env','V2D_STATIC_GRASP_SOURCE_FRAME=34','V2D_GRASP_SEARCH_GENERATIONS=12',
                'V2D_GRASP_PROBE_REFERENCE='+reference,'V2D_REFINED_GRASP_LABEL='+label,
                'V2D_GRASP_USE_REFERENCE_SEED=1','V2D_GRASP_SEARCH_OBJECTS=["mini_sweeper"]',
                'V2D_GRASP_CONSISTENT_INITIAL_MOTORS=1','V2D_GRASP_SEARCH_REJECT_INVALID=1',
                'V2D_PROTECT_OTHER_OBJECTS=1','V2D_GRASP_TRANSLATION_BOUND_M=.045','V2D_GRASP_ROTATION_BOUND_RAD=.6',
                'python','-u','/episode/scripts/diagnose_refined_grasp_search.py'],ROOT/f'logs/{label}.log')
    result=json.loads((ROOT/f'evaluation/{label}.json').read_text());fit=result['best_candidates']['mini_sweeper']
    save(OUT/'anatomical-grasp-search-result.json',result)
    if not fit['stable_hold_and_lift']:
        status('needs_repair','Source-hand-aligned brush grasp remains unstable; training blocked',fit=fit);return
    candidate='track3_pickup_anatomical_candidate'
    if not (ROOT/'research/anatomical-reference-applied.json').exists():
        entry=next(g for g in json.loads((SOURCE/'research/collision-aware-grasps.json').read_text())['grasps'] if g['object']=='mini_sweeper')
        entry.update(wrist_in_object=fit['wrist_in_object'],finger_joints=fit['finger_joints'],physical_pickup_fit=fit)
        save(ROOT/'research/anatomical-fitted-grasp.json',{'grasps':[entry]})
        text=(ROOT/'scripts/apply_pickup_fit.py').read_text()
        text=text.replace('human_motion_data/track3_collision_variant/processed','human_motion_data/'+reference+'/processed')
        text=text.replace('human_motion_data/track3_pickup_candidate/processed','human_motion_data/'+candidate+'/processed')
        text=text.replace('pickup-fitted-grasp.json','anatomical-fitted-grasp.json').replace('pickup-reference-applied.json','anatomical-reference-applied.json')
        text=text.replace('lo=max(0,start-5)','lo=max(0,start-10)')
        (ROOT/'scripts/apply_anatomical_pickup_fit.py').write_text(text)
        docker(['exec',NAME.removesuffix('newton')+'retarget','/isaac-sim/python.sh','-u','/episode/scripts/apply_anatomical_pickup_fit.py'],ROOT/'logs/apply-anatomical-pickup-fit.log')
    shutil.copy2(BASE/'diagnose_episode30_cold_pickup.py',ROOT/'scripts/diagnose_episode30_cold_pickup.py')
    status('running','Testing source-hand-aligned physical pickup from frame0 and continuous brush motion')
    gate_label='anatomical-cold-pickup-gate'
    if not (ROOT/f'evaluation/{gate_label}.json').exists():
        docker(['exec',NAME,'env','V2D_GRASP_PROBE_REFERENCE='+candidate,'V2D_PICKUP_GATE_LABEL='+gate_label,
                'V2D_WRIST_MOTOR_PROFILE={"kp":250,"kd":20}','python','-u','/episode/scripts/diagnose_episode30_cold_pickup.py'],ROOT/'logs/anatomical-cold-pickup-gate.log')
    gate=json.loads((ROOT/f'evaluation/{gate_label}.json').read_text());save(OUT/'anatomical-cold-pickup-gate.json',gate)
    if not gate['pickup_gate_passed']:
        status('needs_repair','Source-hand-aligned brush approach still fails physical pickup; optimizing actual cold approach',cold_pickup=gate)
        return cold_approach_search()
    train_verified_pickup(gate,candidate)

def coupled_reconstruction_retry():
    status('running','Reconstructing brush motion jointly with observed hand pixels and a physically tested grip')
    import numpy as np,pyarrow as pa,pyarrow.parquet as pq
    from scipy.spatial.transform import Rotation
    suffix='processed/sequence_id=episode_000030/robot_name=sharpa_wave/motion.parquet'
    seed_reference='track3_coupled_seed';seed_path=ROOT/'human_motion_data'/seed_reference/suffix
    if not seed_path.exists():
        data=pq.read_table(ROOT/'human_motion_data/track3_pickup_anatomical'/suffix).to_pylist()[0]
        motion=np.load(ROOT/'reconstruction/scene_motion_physical.npz');O=motion['object_to_world'][:,0]
        pos=np.array(data['object_body_position']);q=np.array(data['object_body_wxyz']);pos[35:245,0]=O[35:245,:3,3];q[35:245,0]=Rotation.from_matrix(O[35:245,:3,:3]).as_quat(scalar_first=True)
        data['object_body_position']=pos.tolist();data['object_body_wxyz']=q.tolist();seed_path.parent.mkdir(parents=True,exist_ok=True);pq.write_table(pa.Table.from_pylist([data]),seed_path,compression='zstd')
        stage=ROOT/'human_motion_data'/seed_reference/'reconstructed_stage';stage.mkdir(exist_ok=True);shutil.copy2(ROOT/'human_motion_data/track3_collision_variant/reconstructed_stage/episode_000030_support.usda',stage)
    label='coupled-source127-grasp'
    if not (ROOT/f'evaluation/{label}.json').exists():
        docker(['exec',NAME,'env','V2D_STATIC_GRASP_SOURCE_FRAME=127','V2D_GRASP_SEARCH_GENERATIONS=12',
                'V2D_GRASP_PROBE_REFERENCE='+seed_reference,'V2D_REFINED_GRASP_LABEL='+label,'V2D_GRASP_USE_REFERENCE_SEED=1',
                'V2D_GRASP_SEARCH_OBJECTS=["mini_sweeper"]','V2D_GRASP_CONSISTENT_INITIAL_MOTORS=1','V2D_GRASP_SEARCH_REJECT_INVALID=1',
                'V2D_PROTECT_OTHER_OBJECTS=1','V2D_GRASP_TRANSLATION_BOUND_M=.045','V2D_GRASP_ROTATION_BOUND_RAD=.6',
                'python','-u','/episode/scripts/diagnose_refined_grasp_search.py'],ROOT/f'logs/{label}.log')
    fit=json.loads((ROOT/f'evaluation/{label}.json').read_text())['best_candidates']['mini_sweeper']
    if not fit['stable_hold_and_lift']:
        # Keep the strict physics thresholds. Reuse the independently verified
        # approach-frame grasp of this identical metric brush instead.
        label='anatomical-source34-grasp'
        fit=json.loads((ROOT/f'evaluation/{label}.json').read_text())['best_candidates']['mini_sweeper']
        assert fit['stable_hold_and_lift']
    status('running','Fitting brush silhouette and observed hand jointly on all263 source frames with a physically verified grip')
    with (ROOT/'logs/coupled-source-reconstruction.log').open('a') as log:
        subprocess.run([str(ROOT/'.venv-video/bin/python'),str(BASE/'reconstruct_episode30_coupled_grasp.py'),'--run',str(ROOT),'--grasp-label',label],stdout=log,stderr=subprocess.STDOUT,check=True)
    reconstruction=json.loads((ROOT/'research/coupled-brush-reconstruction.json').read_text())
    if not (OUT/'brush-reconstruction-source-review.mp4').exists():
        subprocess.run([str(ROOT/'.venv-video/bin/python'),str(BASE/'render_episode30_coupled_reconstruction.py'),'--run',str(ROOT)],check=True)
    if reconstruction['mean_source_iou_after']<reconstruction['mean_source_iou_before']-.01:
        status('needs_repair','Joint brush/hand reconstruction worsened source silhouette agreement; training blocked',
               source_iou_before=reconstruction['mean_source_iou_before'],source_iou_after=reconstruction['mean_source_iou_after']);return
    if not (ROOT/'research/coupled-reference-applied.json').exists():
        entry=next(g for g in json.loads((SOURCE/'research/collision-aware-grasps.json').read_text())['grasps'] if g['object']=='mini_sweeper')
        entry.update(wrist_in_object=fit['wrist_in_object'],finger_joints=fit['finger_joints'],physical_pickup_fit=fit)
        save(ROOT/'research/coupled-fitted-grasp.json',{'grasps':[entry]})
        text=(ROOT/'scripts/apply_anatomical_pickup_fit.py').read_text()
        text=text.replace('human_motion_data/track3_pickup_anatomical/processed','human_motion_data/track3_coupled_brush/processed')
        text=text.replace('human_motion_data/track3_pickup_anatomical_candidate/processed','human_motion_data/track3_coupled_pickup_candidate/processed')
        text=text.replace('anatomical-fitted-grasp.json','coupled-fitted-grasp.json').replace('anatomical-reference-applied.json','coupled-reference-applied.json')
        compile(text,'apply_coupled_grasp.py','exec');(ROOT/'scripts/apply_coupled_grasp.py').write_text(text)
        docker(['exec',NAME.removesuffix('newton')+'retarget','/isaac-sim/python.sh','-u','/episode/scripts/apply_coupled_grasp.py'],ROOT/'logs/apply-coupled-grasp.log')
    return cold_approach_search(reference_override='track3_coupled_pickup_candidate',phase_tag='coupled')

def cold_approach_search(reference_override=None,phase_tag='two-hand'):
    status('running','Optimizing both physical grips, closure timing and hand clearance through real pickup from frame0; no object resets',
           progress_log=str(ROOT/'logs/cold-pickup-search.log'),search_generations=12)
    reference=reference_override or 'track3_pickup_source_hand'
    if reference_override is None and not (ROOT/'research/source-hand-grasp-reference.json').exists():
        text=(ROOT/'scripts/apply_anatomical_pickup_fit.py').read_text()
        text=text.replace('human_motion_data/track3_pickup_anatomical_candidate/processed','human_motion_data/'+reference+'/processed')
        text=text.replace('anatomical-reference-applied.json','source-hand-grasp-reference.json')
        text=text.replace("  for i in range(lo,hi+1):", "  anchor=fit['physical_pickup_fit']['source_frame'];raw_anchor=np.eye(4);raw_anchor[:3,3]=w[anchor];raw_anchor[:3,:3]=Rotation.from_quat(Q[anchor],scalar_first=True).as_matrix();hand_delta=np.linalg.inv(raw_anchor)@O[anchor,b]@localbase\n  for i in range(lo,hi+1):")
        text=text.replace('new=O[i,b]@localbase;', 'raw_hand=np.eye(4);raw_hand[:3,3]=w[i];raw_hand[:3,:3]=Rotation.from_quat(Q[i],scalar_first=True).as_matrix();new=raw_hand@hand_delta;')
        (ROOT/'scripts/build_source_hand_grasp.py').write_text(text)
        docker(['exec',NAME.removesuffix('newton')+'retarget','/isaac-sim/python.sh','-u','/episode/scripts/build_source_hand_grasp.py'],ROOT/'logs/build-source-hand-grasp.log')
    shutil.copy2(BASE/'search_episode30_cold_pickup.py',ROOT/'scripts/search_episode30_cold_pickup.py')
    result_path=ROOT/'evaluation/cold-pickup-search.json'
    seed=json.loads(result_path.read_text()) if result_path.exists() and reference_override is None else None
    if seed and not seed.get('parameter_layout'):
        save(ROOT/'research/cold-search-before-two-hand-repair.json',seed)
        shutil.copy2(ROOT/'evaluation/cold-pickup-search-best.npz',ROOT/'research/cold-search-before-two-hand-repair.npz')
    for attempt,profile in enumerate([{'kp':250,'kd':20},{'kp':500,'kd':30}]):
        completed=ROOT/f'evaluation/cold-pickup-search-{phase_tag}-round-{attempt}.json'
        if completed.exists():result=json.loads(completed.read_text())
        else:
            if seed:save(ROOT/'research/cold-approach-search-seed.json',seed)
            docker(['exec',NAME,'env','V2D_COLD_PICKUP_GENERATIONS=12','V2D_COLD_PICKUP_ROUND='+str(attempt+1),
                    'V2D_COLD_PICKUP_PROFILE='+json.dumps(profile),
                    'V2D_COLD_PICKUP_REFERENCE='+reference,
                    *(['V2D_COLD_PICKUP_SEED=research/cold-approach-search-seed.json'] if seed else []),
                    'python','-u','/episode/scripts/search_episode30_cold_pickup.py'],ROOT/'logs/cold-pickup-search.log')
            result=json.loads(result_path.read_text());save(completed,result)
            shutil.copy2(ROOT/'evaluation/cold-pickup-search-best.npz',completed.with_suffix('.npz'))
        save(OUT/'cold-approach-search.json',result)
        seed=result
        if result['best']['cold_pickup_verified']:break
    save(result_path,result)
    if not result['best']['cold_pickup_verified']:
        status('needs_repair','Cold approach search has not produced a verified brush pickup; no new policy published or training started',best=result['best'])
        if phase_tag!='coupled':return coupled_reconstruction_retry()
        # A corrected, physically viable reference can still require closed-loop
        # feedback. Train the actual repair policy from approach on every reset;
        # do not claim the failed open-loop diagnostic was a successful pickup.
        cfg=json.loads((ROOT/'episode-config.json').read_text())
        cfg.update(wrist_motor_profile={'kp':500.,'kd':30.},requires_brush_pickup_validation=True,
                   training_focus={'source_interval':[0,0],'source_frame_count':263,'probability':1.},
                   training_reset_reason='Every training episode starts at frame0, before either object is grasped.',
                   open_loop_cold_pickup_verified=False,
                   training_justification='Closed-loop policy repair after joint source-pixel/hand reconstruction and isolated physical grasp verification; open-loop pickup still fails.')
        save(ROOT/'episode-config.json',cfg)
        import numpy as np
        backup=ROOT/'reconstruction/scene_motion_before_coupled_policy_reference.npz'
        motion_path=ROOT/'reconstruction/scene_motion_physical.npz'
        if not backup.exists():
            shutil.copy2(motion_path,backup);loaded=np.load(motion_path);motion={k:loaded[k].copy() for k in loaded.files}
            motion['object_to_world'][:,0]=np.load(ROOT/'reconstruction/brush_coupled_motion.npz')['object_to_world']
            np.savez_compressed(motion_path,**motion)
        preview=ROOT/'videos/reconstruction-source-review.mp4'
        if not preview.exists():preview.symlink_to(OUT/'brush-reconstruction-source-review.mp4')
        save(ROOT/'research/full-reconstruction-completed.json',{'completed_utc':utc(),'all_vision_stages_rerun':False,
             'method':'Joint brush silhouette and source hand projection fit with an independently tested physical grasp',
             'source_frames':263,'original_calibration_masks_and_metric_mesh_reused':True,'old_policy_reused':False,
             'sam3d_reconstruction_used':False,'open_loop_pickup_verified':False,'learned_pickup_validation_required':True})
        gate={'pickup_gate_passed':False,'isolated_grasp_verified':True,'open_loop_cold_pickup_verified':False,
              'worlds':128,'worlds_with_physical_pickup':0,'best':result['best'],
              'training_is_a_repair_experiment':True,'full_source_motion_verified':False}
        return train_verified_pickup(gate,reference)
    # Bake only verified hand motor commands into a new training reference.
    candidate='track3_pickup_cold_candidate'
    if not (ROOT/'research/verified-cold-reference.json').exists():
        text=(ROOT/'scripts/apply_anatomical_pickup_fit.py').read_text()
        text=text.replace('human_motion_data/track3_pickup_anatomical/processed','human_motion_data/'+reference+'/processed')
        text=text.replace('human_motion_data/track3_pickup_anatomical_candidate/processed','human_motion_data/'+candidate+'/processed')
        start=text.index(" for fit in [g for g in fits if g['side']==side]:")
        end=text.index(' # Recompute all FK',start)
        block=''' cold=json.loads((R/'evaluation/cold-pickup-search.json').read_text());values=np.array(cold['best']['parameters']);F=cold['parameter_layout']['right_finger_end']
 if side=='left':
  flex=np.array([name.endswith(('_MCP_FE','_PIP','_DIP','_IP')) for name in cold['left_finger_names']]);limits=cold['left_finger_limits']
  for i in range(T):
   blend=float(np.clip(min((i-8)/6.,(253-i)/6.),0.,1.));w[i]+=blend*values[F:F+3];f[i,flex]*=1+blend*(values[F+3]-1);f[i]=np.clip(f[i],limits['lower'],limits['upper'])
 if side=='right':
  opened=values[9:F].copy()
  for j,n in enumerate(c['right_finger_joint_names'] if 'right_finger_joint_names' in c else c['right_robot_joint_names']):
   if n.endswith(('_MCP_FE','_PIP','_DIP','_IP')):opened[j]=0.
  for i in range(T):
   blend=float(np.clip(min((i-20)/10.,(253-i)/8.),0.,1.));w[i]+=blend*values[:3];w[i,2]+=values[8]*max(0.,1-abs(i-25)/9.)
   Q[i]=(Rotation.from_quat(Q[i],scalar_first=True)*Rotation.from_rotvec(blend*values[3:6])).as_quat(scalar_first=True)
   closing=float(np.clip((i-(values[6]-values[7]/2))/values[7],0.,1.))*float(np.clip((253-i)/8.,0.,1.))
   f[i]=(1-closing)*opened+closing*values[9:F]
  report.append({'verified_physical_cold_pickup':True,'parameters':values.tolist(),'object_poses_modified':False})
'''
        # Name the finger joints from the explicitly saved physical fit.
        names=json.loads((ROOT/'evaluation/anatomical-source34-grasp.json').read_text())['best_candidates']['mini_sweeper']['finger_joint_names']
        block=block.replace("c['right_finger_joint_names'] if 'right_finger_joint_names' in c else c['right_robot_joint_names']",repr(names))
        text=text[:start]+block+text[end:]
        text=text.replace('anatomical-reference-applied.json','verified-cold-reference.json')
        (ROOT/'scripts/bake_verified_cold_reference.py').write_text(text)
        docker(['exec',NAME.removesuffix('newton')+'retarget','/isaac-sim/python.sh','-u','/episode/scripts/bake_verified_cold_reference.py'],ROOT/'logs/bake-verified-cold-reference.log')
    gate={'pickup_gate_passed':True,'method':'Physics-based cold approach search','worlds':128,
          'worlds_with_physical_pickup':result['history'][-1]['verified_cold_pickups'],'best':result['best'],
          'policy_inference':False,'full_source_motion_verified':False}
    cfg=json.loads((ROOT/'episode-config.json').read_text())
    cfg['wrist_motor_profile']=result['wrist_motor_profile']
    cfg['training_focus']={'source_interval':[0,0],'source_frame_count':263,'probability':1.}
    cfg['training_reset_reason']='Every learning episode starts before pickup; later held-object resets cannot hide a failed approach.'
    save(ROOT/'episode-config.json',cfg)
    train_verified_pickup(gate,candidate)

def main():
    prepare()
    if (ROOT/'coherent-input-checksums.json').exists():
        # Frozen training inputs must not enter reconstruction again on resume.
        cfg=json.loads((ROOT/'episode-config.json').read_text())
        candidate=cfg['training_reference']
        assert (ROOT/'human_motion_data'/candidate/'processed/sequence_id=episode_000030/robot_name=sharpa_wave/motion.parquet').exists()
        previous=json.loads((OUT/'status.json').read_text())
        gate=previous.get('cold_pickup',json.loads((OUT/'cold-pickup-gate.json').read_text()))
        return train_verified_pickup(gate,candidate)
    status('running','Physical brush pickup search: source frame34, 128 independent worlds, no object assistance')
    label='pickup-repair-source34-handle'
    if not (ROOT/f'evaluation/{label}.json').exists():
        docker(['exec',NAME,'env','V2D_STATIC_GRASP_SOURCE_FRAME=34','V2D_GRASP_SEARCH_GENERATIONS=12',
                'V2D_REFINED_GRASP_LABEL='+label,'V2D_GRASP_SEED_FIT=research/previous-handle-grasp-seed.json',
                'V2D_GRASP_SEARCH_OBJECTS=["mini_sweeper"]','V2D_GRASP_CONSISTENT_INITIAL_MOTORS=1',
                'V2D_GRASP_SEARCH_REJECT_INVALID=1','V2D_PROTECT_OTHER_OBJECTS=1',
                'V2D_GRASP_TRANSLATION_BOUND_M=.045','V2D_GRASP_ROTATION_BOUND_RAD=.6',
                'python','-u','/episode/scripts/diagnose_refined_grasp_search.py'],ROOT/f'logs/{label}.log')
    result=json.loads((ROOT/f'evaluation/{label}.json').read_text())
    fit=result['best_candidates']['mini_sweeper']
    save(OUT/'pickup-search-result.json',result)
    if not fit['stable_hold_and_lift']:
        status('needs_repair','No physically stable brush pickup found; policy training blocked',fit=fit)
        return
    status('running','Applying physically tested brush grip; rebuilding robot FK and source-timed contact labels',fit=fit)
    candidate='track3_pickup_candidate'
    marker=ROOT/'research/pickup-reference-applied.json'
    if not marker.exists():
        original=json.loads((SOURCE/'research/collision-aware-grasps.json').read_text())
        entry=next(g for g in original['grasps'] if g['object']=='mini_sweeper')
        entry.update(wrist_in_object=fit['wrist_in_object'],finger_joints=fit['finger_joints'],physical_pickup_fit=fit)
        save(ROOT/'research/pickup-fitted-grasp.json',{'grasps':[entry]})
        text=(BASE/'apply_collision_grasps.py').read_text()
        text=text.replace('human_motion_data/track3_coherent_variant/processed','human_motion_data/track3_collision_variant/processed')
        text=text.replace("R/'research/collision-aware-grasps.json'","R/'research/pickup-fitted-grasp.json'")
        text=text.replace("out=R/f'human_motion_data/track3_collision_variant/processed",f"out=R/f'human_motion_data/{candidate}/processed")
        text=text.replace("R/'research/collision-aware-applied-grasps.json'","R/'research/pickup-reference-applied.json'")
        (ROOT/'scripts/apply_pickup_fit.py').write_text(text)
        retarget=NAME.removesuffix('newton')+'retarget'
        if _docker(['ps','-a','--filter',f'name=^{retarget}$','--format','{{.Names}}']).stdout.strip()!=retarget:
            info=json.loads(_docker(['inspect','v2d-rebuild23-episode30-retarget']).stdout)[0]
            cmd=['run','-d','--gpus','all','--runtime','nvidia','--ipc','host','--name',retarget]
            for mount in info['Mounts']:
                cmd+=['-v',(str(ROOT) if mount['Destination']=='/episode' else mount['Source'])+':'+mount['Destination']+('' if mount['RW'] else ':ro')]
            cmd+=['--entrypoint','/bin/sleep',info['Image'],'infinity'];_docker(cmd)
        else:_docker(['start',retarget])
        docker(['exec',retarget,'/isaac-sim/python.sh','-u','/episode/scripts/apply_pickup_fit.py'],ROOT/'logs/apply-pickup-fit.log')
    shutil.copy2(BASE/'diagnose_episode30_cold_pickup.py',ROOT/'scripts/diagnose_episode30_cold_pickup.py')
    status('running','Testing physical brush pickup from source frame0 and the full continuous source trajectory; no resets or object assistance')
    if not (ROOT/'evaluation/cold-pickup-gate.json').exists():
        docker(['exec',NAME,'env','V2D_GRASP_PROBE_REFERENCE='+candidate,'V2D_WRIST_MOTOR_PROFILE={"kp":250,"kd":20}',
                'python','-u','/episode/scripts/diagnose_episode30_cold_pickup.py'],ROOT/'logs/cold-pickup-gate.log')
    gate=json.loads((ROOT/'evaluation/cold-pickup-gate.json').read_text());save(OUT/'cold-pickup-gate.json',gate)
    if not gate['pickup_gate_passed']:
        status('needs_repair','Static brush grasp passes, but approach/pickup from frame0 still fails; training blocked',cold_pickup=gate)
        return anatomical_retry()
    train_verified_pickup(gate,candidate)

def train_verified_pickup(gate,candidate):
    cfg=json.loads((ROOT/'episode-config.json').read_text())
    cfg.update(training_reference=candidate,reference_review_pending=False,all_frames_visually_reviewed=False,
               source_frames_manually_reviewed=[0,10,14,24,30,34,38,42,46,50,60,80,100,120,140,160,180,200,220,240,245,250,262],
               physical_pickup_gate='evaluation/cold-pickup-gate.json',sam3d_reconstruction_used=False)
    save(ROOT/'episode-config.json',cfg)
    saved=list((ROOT/'flash_chord_coherent_actor_transfer').glob('state_*.safetensors'))
    phase=('Resuming saved policy training' if saved else 'Starting fresh policy training')
    status('running',phase+' with every episode starting at frame0; learned pickup remains unverified',cold_pickup=gate)
    registry=BATCH/'improvements_20261021/active-runs.json';entries=json.loads(registry.read_text());entries['30']=ROOT.name;save(registry,entries)
    import rerun_four_episodes_20261023 as pipeline
    pipeline.root=lambda episode:ROOT
    pipeline.OUT=OUT;pipeline.EPISODES=[30];pipeline.VARIANT='pickup_repair_20261026'
    original_better=pipeline.better
    def verified_better(candidate,previous):
        from assess_episode30_pickup import assess_file
        states=Path(candidate['recorded_states'])
        if str(states).startswith('/episode/'):states=ROOT/str(states).removeprefix('/episode/')
        elif not states.is_absolute():states=ROOT/states
        check=assess_file(ROOT,states);check.update(states=str(states),states_sha256=digest(states));save(ROOT/'research/learned-policy-pickup-gate.json',check)
        return check['passes'] and original_better(candidate,previous)
    pipeline.better=verified_better
    pipeline.train_and_export(30)
    assert digest(SOURCE/'deliverables/manifest.json')==json.loads((ROOT/'research/pickup-isolation.json').read_text())['source_manifest_sha256']
    status('completed','Fresh policy and full-source rollout checked; exact human matching remains unverified',cold_pickup=gate)

if __name__=='__main__':
    try:
        from episode_execution_guard import episode_slot
        with episode_slot(ROOT,30):main()
    except Exception:
        status('failed','Physical pickup repair failed; previous videos preserved',error=traceback.format_exc());raise
