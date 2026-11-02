"""Run one source-diagnosed, isolated corrective experiment for any evaluation clip."""
import argparse
import fcntl
import importlib.util
import json
import os
from pathlib import Path
import shlex
import shutil
import stat
import subprocess
import traceback
import types
import time
from improvement_common import BASE,BATCH,save,utc,digest,better
from human_match_criteria import assess
os.environ['PYTHONPATH']=str(BASE)+os.pathsep+os.environ.get('PYTHONPATH','')

PY=BASE/'runs/episode_000005_newton_20261015/.venv-video/bin/python'
ROSTER=[1,3,5,6,8,10,14,15,16,19,20,25,28,30,32,33,35,36,37,38]
p=argparse.ArgumentParser();p.add_argument('--request',type=Path,required=True);p.add_argument('--prepare-only',action='store_true');args=p.parse_args()
job=json.loads(args.request.read_text());ep=job['episode'];generation=job['generation'];assert ep in ROSTER
support_path=BATCH/'requested_episodes_20_30/episode20-object-support.json'
if ep==20 and job.get('require_full_sequence_pickup') and support_path.exists():
    support=json.loads(support_path.read_text())
    if support.get('physical_stability_verified') and digest(support['physical_report'])==support['physical_report_sha256']:
        settings=dict(support['refinement_settings']);settings['rest_upright_mesh_sha256']=support['mesh_sha256']
        job.setdefault('object_pixel_refinement',{})['water_pitcher']=settings
        save(args.request,job)
source=Path(job['source_run']);recsource=BASE/'runs/episode_000001_newton_v5_20261014' if ep==1 else source
out=BATCH/f'improvements_20261023_match_g{generation:03d}'
root=BASE/f'runs/episode_{ep:06d}_newton_human_match_20261023_g{generation:03d}'
name=f'v2d-rebuildmatch23g{generation:03d}-episode{ep}-newton'
module=types.ModuleType('human_match_pipeline');module.__dict__.update(__file__=str(BASE/'rerun_four_episodes_20261023.py'),MATCH_GENERATION=generation)
code=(BASE/'rerun_four_episodes_20261023.py').read_text().replace('20261023+ep','20261023+ep+997*MATCH_GENERATION')
exec(compile(code,module.__file__,'exec'),module.__dict__)
w=module;w.OUT=out;w.VARIANT=f'human_match_20261023_g{generation:03d}';w.EPISODES=[ep]

def supported_intervals(cfg,body):
    held={i for g in cfg['grasps'] if g['object']==body for i in range(g['start'],g['end']+1)}
    spans=[]
    for i in range(cfg['frames']):
        if i in held:continue
        if spans and i==spans[-1][1]+1:spans[-1][1]=i
        else:spans.append([i,i])
    return spans

def prepare():
    if (root/'episode-config.json').exists() and (root/'research/preserved-input-release.json').exists():
        if not (root/'coherent-input-checksums.json').exists():
            cfg=json.loads((root/'episode-config.json').read_text())
            cfg.update(requires_no_penetration_for_publication=True,require_nonpenetrating_reference_before_training=True,
                       grasp_search_require_nonpenetrating_initial_pose=True)
            save(root/'episode-config.json',cfg)
        return
    root.mkdir(exist_ok=True);out.mkdir(exist_ok=True)
    for folder in ['scripts','research','logs','evaluation','videos','deliverables','reconstruction','human_motion_data','flash_chord_coherent_actor_transfer']:(root/folder).mkdir(exist_ok=True)
    cfgpath=source/'episode-config.json'
    if cfgpath.exists():cfg=json.loads(cfgpath.read_text())
    else:
        cfg=json.loads((BASE/'runs/episode_000003_newton_batch_20261015/episode-config.json').read_text())
        cfg={k:cfg[k] for k in ['resolution','fps']}
        cfg.update(episode=ep,sequence=f'episode_{ep:06d}',objects=['white_pot','white_pot_lid'] if ep==1 else ['white_pot','wooden_spoon'],
                   frames=335 if ep==1 else 288,
                   grasps=([{'side':'left','object':0,'start':36,'end':302},{'side':'right','object':1,'start':36,'end':97},{'side':'right','object':1,'start':247,'end':302}] if ep==1 else
                           [{'side':'left','object':0,'start':22,'end':249},{'side':'right','object':1,'start':37,'end':268}]))
    for key in ['physical_grasp_refinement','predecomposed_object_collisions','training_seed_checkpoint_run','contact_full_state_seed','adaptive_single_prefix','wrist_side_motor_profiles','reconstruction_reused_from']:
        cfg.pop(key,None)
    cfg.update(run=str(root),baseline_run=job['baseline_run'],training_variant=w.VARIANT,native_container=name,
               source=str(BASE/f'datasets/video_to_data_challenge/track_3/evaluation/videos/chunk-000/observation.images.ego_cam_a/episode_{ep:06d}.mp4'),
               training_reference='track3_collision_variant',training_initial_counter=0,training_budget=100000256,training_world_count=512,
               training_curriculum='fresh_rebuild',training_compute_dtype='float32',physics_substeps=20,physics_integration_hz=400,
               physics_nconmax=2048,physics_njmax=4096,wrist_motor_profile={'kp':250.,'kd':20.},training_numeric_failure_guard=True,
               frame_cache_format='png',reference_review_pending=True,all_frames_visually_reviewed=False,
               all_source_frames_decoded_and_automatically_checked=False,requires_verified_improvement_for_publication=True,
               match_output=str(out),audit_roster=ROSTER,match_request=str(args.request),
               improvement_experiment='Source-video mismatch corrective experiment; new policy initialization and seed')
    cfg['requires_no_penetration_for_publication']=True
    if job.get('validated_video_pipeline'):
        cfg.update(validated_video_pipeline=True,source_only_reconstruction=True,
                   training_budget=int(job.get('training_budget',12_800_000)),
                   training_round_increment=12_800_000,submission_mesh_face_budget=3600,
                   public_validation_manifest=job['public_validation_manifest'],
                   time_advance_frames=float(job.get('time_advance_frames',0.)),
                   official_evaluation_required=True)
    if job.get('experimental_training'):
        cfg.update(experimental_training=True,experimental_only=True,
                   experimental_admission=job['experimental_admission'],
                   strict_completion_granted=False)
    if job.get('own_geometry_required'):
        cfg.update(own_geometry_required=True,dataset_split='evaluation',
                   own_training_continuation_allowed=job.get('own_training_continuation_allowed',False),
                   own_training_maximum_steps=job.get('own_training_maximum_steps',cfg['training_budget']))
    if job.get('fast_submission'):cfg.update(fast_submission=True,official_recorded_only=True)
    if job.get('actor_transfer'):
        transfer=dict(job['actor_transfer']);seed=root/'transfer_seed/policy.safetensors'
        seed.parent.mkdir(exist_ok=True);shutil.copy2(transfer['source_checkpoint'],seed)
        assert digest(seed)==transfer['source_checkpoint_sha256'],'Actor-transfer source changed'
        transfer['checkpoint']='/episode/transfer_seed/policy.safetensors'
        cfg['actor_transfer']=transfer
    cfg['require_nonpenetrating_reference_before_training']=True
    cfg['grasp_search_require_nonpenetrating_initial_pose']=True
    options=job.get('worst_windows') or [{'start':0,'end':min(cfg['frames']-1,50)}]
    window=options[(generation-1)%len(options)]
    cfg['training_focus']={'source_interval':[max(0,window['start']-5),min(cfg['frames']-1,window['end']+5)],'source_frame_count':cfg['frames'],'probability':.35}
    if job.get('require_full_sequence_pickup'):
        cfg['training_focus']={'source_interval':[0,0],'source_frame_count':cfg['frames'],'probability':1.}
        cfg['requires_strict_source_match_for_publication']=True
        if ep==20:
            cfg['require_physical_holds_before_training']=True
            cfg['training_focus']['probability']=0.0
            cfg['training_reset_reason']='Restore native staged source-frame sampling for learning; all acceptance rollouts still start at0 with continuous source timing and zero object assistance.'
    if ep in [19,30]:cfg['maximum_floor_projection_m']=.04
    if job.get('validated_video_pipeline'):cfg.pop('training_focus',None)
    cfg['source_support_intervals']=cfg.get('source_support_intervals',cfg.get('support_intervals',{}))
    if 'wooden_spoon' in cfg['objects']:
        cfg['source_support_intervals'].setdefault('wooden_spoon',supported_intervals(cfg,cfg['objects'].index('wooden_spoon')))
    cfg['support_intervals']=cfg['source_support_intervals']
    save(root/'episode-config.json',cfg)
    if not (root/'.venv-video').is_symlink():(root/'.venv-video').symlink_to(PY.parent.parent,target_is_directory=True)
    if not (root/'models').is_symlink():(root/'models').symlink_to(BASE/'runs/episode_000005_newton_20261015/models',target_is_directory=True)
    review=recsource/'research/source-video-review.json'
    if review.exists():shutil.copy2(review,root/'research/source-video-review.json')
    else:
        legacy=json.loads((BATCH/f'human_video_recheck_20261021/episode_{ep:06d}/source-phase-annotations.json').read_text())
        save(root/'research/source-video-review.json',{'frame_count':cfg['frames'],'grasps':cfg['grasps'],'phases':legacy['phases'],'all_frames_visually_reviewed':False})
    shutil.copy2(recsource/'reconstruction/sam2_prompts.json',root/'reconstruction/sam2_prompts.json')
    save(root/'research/rebuild-input-provenance.json',{'updated_utc':utc(),'source_videos':[{'path':str(BASE/f'datasets/video_to_data_challenge/track_3/evaluation/videos/chunk-000/observation.images.ego_cam_{cam}/episode_{ep:06d}.mp4'),'sha256':digest(BASE/f'datasets/video_to_data_challenge/track_3/evaluation/videos/chunk-000/observation.images.ego_cam_{cam}/episode_{ep:06d}.mp4')} for cam in 'abc'],
         'old_estimated_camera_object_hand_poses_copied':False,'old_masks_or_depth_copied':False,
         'old_actor_checkpoint_copied':bool(job.get('actor_transfer')),'old_optimizer_copied':False,'trigger':job})
    from preserved_release import original_artifacts
    watched=original_artifacts(source,ep)
    save(root/'research/preserved-input-release.json',watched)

def verify(_):
    for path,expected in json.loads((root/'research/preserved-input-release.json').read_text()).items():assert digest(path)==expected,path
w.verify_preserved=verify
original_attach=w.attach
w.attach=lambda r,row:original_attach(r,row) if (Path(r)/'episode-config.json').exists() else row
original_better=w.better
w.better=lambda candidate,previous:False if previous is None or job.get('experimental_training') else original_better(candidate,previous)

def container(r,episode,kind):
    container_name=name.removesuffix('newton')+kind
    exists=subprocess.run(['sg','docker','-c',shlex.join(['docker','inspect',container_name])],capture_output=True,text=True)
    if exists.returncode==0:
        if not json.loads(exists.stdout)[0]['State']['Running']:w._docker(['start',container_name])
        return container_name
    template=json.loads(w._docker(['inspect',f'v2d-batch-episode19-{kind}']).stdout)[0]
    cmd=['run','-d','--gpus','all','--runtime','nvidia','--ipc','host','--name',container_name]
    for mount in template['Mounts']:cmd+=['-v',(str(r) if mount['Destination']=='/episode' else mount['Source'])+':'+mount['Destination']+('' if mount['RW'] else ':ro')]
    for device in sorted([*Path('/dev').glob('nvidia*'),*Path('/dev/nvidia-caps').glob('*')]):
        if stat.S_ISCHR(device.stat().st_mode):cmd+=['--device',str(device)]
    cmd+=['-e','PYGLET_HEADLESS=true','-e','XLA_PYTHON_CLIENT_PREALLOCATE=false']
    if kind=='newton':cmd+=['--workdir','/workspace']
    cmd+=['--entrypoint','/bin/sleep',template['Image'],'infinity'];w._docker(cmd,r/f'logs/create-{kind}.log');return container_name
w.make_container=container
original_command=w.command
def command(r,argv,label):
    argv=list(argv)
    if job.get('approved_source_cad_run') and label in ['rebuild-source-constraints','rebuild-supported-spoon']:
        # CAD/source poses have already passed their own per-frame gate.
        # Robot retargeting must follow them rather than refitting the objects.
        return
    if str(argv[1]).endswith('review_reconstructed_source_20261023.py'):
        cfg=json.loads((r/'episode-config.json').read_text())
        save(r/'research/source-review-verified.json',{'updated_utc':utc(),'method':'Every original frame decoded and scored; manual full-frame review remains pending for this new trial.','source_frames_reviewed':[],
             'original_video_hashes_verified':True,'new_policy_manually_verified':False})
        cfg.update(all_frames_visually_reviewed=False,visual_review_record='research/source-review-verified.json');save(r/'episode-config.json',cfg);return
    if str(argv[1]).endswith('rebuild_source_constraints_20261022.py') and ep not in [19,20,30,33]:
        text=(BASE/'rebuild_source_constraints_20261022.py').read_text()
        text=text.replace("subprocess.run([str(r/'.venv-video/bin/python'),str(r.parents[1]/'repair_anatomical_hands_20261022.py'),'--run',str(r)],check=True)","# Preserve generic anatomical hand tracking and actual handover identities.")
        target=r/'scripts/rebuild_generic_source_constraints.py';target.write_text(text);argv[1]=target
    result=original_command(r,argv,label)
    if label=='rebuild-source-constraints':
        original_command(r,[PY,BASE/'refit_visible_objects_20261023.py','--run',r,'--request',args.request],'refine-visible-source-pixels')
    return result
w.command=command
original_install=w.install
def install(r):
    original_install(r)
    cfg=json.loads((r/'episode-config.json').read_text())
    if 'wooden_spoon' in cfg['objects'] and ep!=33 and not job.get('approved_source_cad_run'):
        text=(BASE/'repair_supported_spoon_20261022.py').read_text().replace("assert c['episode'] == 33","assert 'wooden_spoon' in c['objects']")
        text=text.replace("intervals = [[0, 56], [289, c['frames'] - 1]]","intervals = c['source_support_intervals']['wooden_spoon']")
        target=r/'scripts/repair_generic_supported_spoon.py';target.write_text(text);original_command(r,[PY,target,'--run',r],'rigid-supported-spoon')
w.install=install

def main():
    prepare();out.mkdir(exist_ok=True)
    requested_hashes=job.get('execution_input_hashes',{})
    if requested_hashes:
        actual_hashes={filename:digest(BASE/filename) for filename in requested_hashes}
        assert actual_hashes==requested_hashes,'Queued scientific code changed before episode execution; schedule a fresh generation'
        save(root/'research/job-execution-inputs.json',dict(updated_utc=utc(),source_code_hashes=actual_hashes,
             generation=generation,request=str(args.request),resumed_frozen_inputs=(root/'coherent-input-checksums.json').exists()))
    if job.get('fast_submission'):
        from fast_policy_execution import run as fast_execute
        result=fast_execute(root,job,container,w.status)
        verify(ep)
        save(args.request.with_suffix('.result.json'),dict(result,all_pipeline_stages_executed=True,episode_done=False,strict_completion_granted=False))
        return
    if job.get('approved_source_cad_run'):
        from install_source_cad_reference import install as install_approved
        install_approved(root,job['approved_source_cad_run'])
    save(root/'research/improvement-status.json',{'episode':ep,'status':'reconstructing','run':str(root),'updated_utc':utc()})
    ids=w._docker(['ps','-q']).stdout.split()
    if ids:
        for info in json.loads(w._docker(['inspect',*ids]).stdout):
            if not any(m.get('Source')==str(root) for m in info.get('Mounts',[])):continue
            top=w._docker(['top',info['Id'],'-eo','pid,args']).stdout
            if '/episode/scripts/train_coherent_actor_transfer.py' not in top:w._docker(['stop','-t','10',info['Id']])
    save(out/'phase-status.json',{'phase':'full_reconstruction_and_training','updated_utc':utc(),'episodes':{},'requested_episodes':[ep]})
    registry=BATCH/'improvements_20261021/active-runs.json';entries=json.loads(registry.read_text());entries[str(ep)]=root.name;save(registry,entries)
    w.status(ep,'Reconstructing from original human video after strict correspondence audit')
    def run_stage(function,label):
        for attempt in range(3):
            try:return function(ep)
            except subprocess.CalledProcessError:
                if attempt==2:raise
                save(root/'research/automatic-stage-retry.json',{'updated_utc':utc(),'stage':label,
                     'attempt_failed':attempt+1,'next_attempt':attempt+2,'saved_stage_outputs_and_checkpoints_preserved':True})
                w.status(ep,f'Automatically retrying {label}; completed work and checkpoints retained')
                time.sleep(30*(attempt+1))
    run_stage(w.rebuild,'reconstruction')
    native=container(root,ep,'newton')
    while not (root/'flash_chord_coherent_actor_transfer/TRAINING_COMPLETED').exists():
        if '/episode/scripts/train_coherent_actor_transfer.py' not in w._docker(['top',native,'-eo','pid,args']).stdout:break
        w.status(ep,'Adopting the existing learner after continuous-loop recovery');time.sleep(15)
    from episode_completion_gate import completion
    rounds=3 if job.get('own_training_continuation_allowed') or (job.get('validated_video_pipeline') and not job.get('experimental_training')) else 1
    for training_round in range(rounds):
        if requested_hashes:
            helper_names=['optimize_native_grasps.py','native_pinch_contacts.py','grasp_physics_checks.py',
                          'rigid_pose_collision_audit.py','apply_collision_transitions.py']
            helper_hashes={filename:digest(root/'scripts'/filename) for filename in helper_names}
            assert all(helper_hashes[name]==requested_hashes[name] for name in helper_names), 'Prepared native helpers do not match the requested scientific code'
            save(root/'research/pretraining-helper-code.json',dict(updated_utc=utc(),sha256=helper_hashes,
                 requested_hashes_matched=True))
        run_stage(w.train_and_export,'policy training, rollout and frame audit')
        report=json.loads((out/f'human_review/episode_{ep:06d}/report.json').read_text())
        frames=json.loads((out/f'human_review/episode_{ep:06d}/frame-metrics.json').read_text())
        strict=assess(report,frames);strict.update(updated_utc=utc(),run=str(root),links=report['links'])
        strict['source_frame_playback']=bool(job.get('validated_video_pipeline'))
        if job.get('experimental_training'):strict.update(experimental_only=True,strict_completion_granted=False)
        selected=json.loads((root/'final-controller-selection.json').read_text())['selected']
        states=Path(selected['recorded_states'])
        if str(states).startswith('/episode/'):states=root/str(states).removeprefix('/episode/')
        elif not states.is_absolute():states=root/states
        strict['states_sha256']=digest(states)
        strict['original_video_sha256']=report['source_video_sha256']
        if json.loads((root/'episode-config.json').read_text()).get('post_training_grasp_acceptance_required'):
            checkpoint=Path(selected['checkpoint'])
            if str(checkpoint).startswith('/episode/'):checkpoint=root/str(checkpoint).removeprefix('/episode/')
            elif not checkpoint.is_absolute():checkpoint=root/checkpoint
            checkpoint_sha=digest(checkpoint)
            acceptance=root/f'evaluation/official/{checkpoint_sha[:16]}/grasp-acceptance.json'
            strict.update(post_training_grasp_acceptance_required=True,checkpoint_sha256=checkpoint_sha,
                post_training_grasp_acceptance=json.loads(acceptance.read_text()) if acceptance.exists() else {},
                official_render_binding=json.loads((root/'evaluation/official-render-binding.json').read_text()))
        save(root/'research/strict-human-match.json',strict)
        locally_ready=False
        if job.get('own_geometry_required'):
            from own_submission_queue import eligible as eligible_for_submission
            try:eligible_for_submission(root);locally_ready=True
            except (ValueError,AssertionError,OSError,KeyError):pass
        if completion(strict)['episode_done'] or locally_ready or training_round==rounds-1 or not job.get('validated_video_pipeline') or (job.get('experimental_training') and not job.get('own_training_continuation_allowed')):break
        from training_continuation import extend
        increment=int(job.get('own_training_round_increment',12_800_000))
        target=extend(root,increment=increment)
        w.status(ep,f'Continue the same frozen-reference policy; target {target} environment steps')
    if ep==1 and not job.get('experimental_training'):
        legacy=json.loads((BATCH/'all20_evaluation/episode_000001-release.json').read_text())
        result_path=out/f'episode_{ep:06d}-fresh-result.json';result=json.loads(result_path.read_text())
        cohort=json.loads((root/'evaluation/final-metrics.json').read_text())['metrics']['chord_sr']
        if legacy.get('selected') is None and strict['passes_strict_local_checks'] and cohort>=legacy['quality']['completion_rate_against_video_estimate']-.0625:
            result['measured_improvement']=True;save(result_path,result);save(root/'research/improvement-status.json',result)
    from episode_completion_gate import completion
    gate=completion(strict)
    save(args.request.with_suffix('.result.json'),{'updated_utc':utc(),'episode':ep,'run':str(root),'strict_check':strict,'all_pipeline_stages_executed':True,'episode_done':gate['episode_done'],'advance_allowed':gate['advance_allowed'],'exact_human_match_verified':False})
    w.status(ep,'Motion verified for every object and source frame' if gate['episode_done'] else 'Attempt finished; episode remains unfinished: object motion is not exactly verified',state='completed' if gate['episode_done'] else 'needs_motion_improvement',episode_done=gate['episode_done'],completion_gate=gate)
    save(out/'phase-status.json',{'phase':'completed','updated_utc':utc(),'requested_episodes':[ep],'exact_human_imitation_verified':False})
    verify(ep)

if __name__=='__main__':
    root.parent.mkdir(exist_ok=True)
    with (args.request.with_suffix('.lock')).open('a') as lock:
        fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
        try:
            if args.prepare_only:prepare();print('Prepared isolated input configuration',root,flush=True)
            else:
                from episode_execution_guard import episode_slot
                with episode_slot(root,ep):main()
        except Exception:
            error=traceback.format_exc();save(args.request.with_suffix('.error.json'),{'updated_utc':utc(),'error':error})
            if not args.prepare_only and (root/'episode-config.json').exists():w.status(ep,'Continuous source-match experiment needs repair',state='failed',error=error)
            raise
