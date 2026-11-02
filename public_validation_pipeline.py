"""Public video-only reconstruction, followed by separate Vicon diagnostics.

This uses the existing serial queue service. Public object poses are loaded only
by the scorer after reconstruction; they never initialise the image tracker.
"""
import json
import os
from pathlib import Path
import shutil
import subprocess
import time
import traceback
import math
from itertools import combinations

from improvement_common import BASE,BATCH,save,utc,digest
from episode_execution_guard import episode_slot,register_episode_child
from public_reference_selection import select_attempt,latest_cad_attempt

OUT=BATCH/'focused_episodes_20261028';VAL=OUT/'pipeline_validation_20261029'
PY=BASE/'runs/episode_000005_newton_20261015/.venv-video/bin/python'
ROOT=BASE/'runs/public_episode_000000_pipeline_20261029'
EPISODE=0
SOURCE_REVISION=''


def configure(episode):
    global ROOT,EPISODE,SOURCE_REVISION
    EPISODE=int(episode)
    SOURCE_REVISION=read(OUT/'frontend-revisions.json',{}).get(str(EPISODE),'')
    suffix=read(OUT/'validated-pipeline-policy.json',{}).get('source_run_suffix','')
    ROOT=BASE/f'runs/public_episode_{EPISODE:06d}_pipeline_20261029{suffix}{SOURCE_REVISION}'


def read(path,default=None):
    try:return json.loads(path.read_text())
    except (ValueError,OSError):return default


def distinct_board_pair(detections):
    """Do not initialise two board IDs from nested or near-identical boxes."""
    pairs=[]
    for first,second in combinations(detections,2):
        a,b=first['box'],second['box']
        areas=[max(0.,r['x1']-r['x0'])*max(0.,r['y1']-r['y0']) for r in (a,b)]
        if min(areas)<=0:continue
        overlap=max(0.,min(a['x1'],b['x1'])-max(a['x0'],b['x0']))*max(0.,min(a['y1'],b['y1'])-max(a['y0'],b['y0']))
        if overlap/(sum(areas)-overlap)>.25 or overlap/min(areas)>.5:continue
        pair=sorted([first,second],key=lambda r:(r['box']['x0']+r['box']['x1'])/2)
        pairs.append((sum(float(r.get('confidence',0)) for r in pair),pair))
    return max(pairs,key=lambda p:p[0])[1] if pairs else None


def copy_measurement(source,target):
    """Reuse immutable inputs; Docker-owned files may prohibit hard linking."""
    source=Path(source);target=Path(target)
    if target.exists():return str(target)
    try:os.link(source,target)
    except OSError:shutil.copy2(source,target)
    return str(target)


def validate_distinct_detection_prompts(prompts):
    """Reject two object IDs initialised on the same observed detector box."""
    for first,second in combinations(prompts,2):
        if (first['object_id']==second['object_id'] or
                first['frame_index']!=second['frame_index'] or
                'box' not in first or 'box' not in second):continue
        a,b=first['box'],second['box']
        areas=[max(0.,box['x1']-box['x0'])*max(0.,box['y1']-box['y0']) for box in (a,b)]
        overlap=max(0.,min(a['x1'],b['x1'])-max(a['x0'],b['x0']))*max(0.,min(a['y1'],b['y1'])-max(a['y0'],b['y0']))
        union=sum(areas)-overlap
        if union>0 and overlap/union>.9:
            raise RuntimeError('Near-identical source detector boxes initialise different object IDs; source-image prompt review needed')


def references_passed(results,episodes,minimum_auc,source_revisions=None):
    if set(results)!={str(ep) for ep in episodes}:return False
    for episode,result in results.items():
        if source_revisions is not None and result.get('source_revision','')!=source_revisions.get(episode,''):return False
        metrics=result.get('best_reference_metrics',result.get('refined_metrics',{}))
        auc=metrics.get('AUC');mp=metrics.get('MP-SR')
        if not isinstance(auc,(int,float)) or not math.isfinite(auc) or auc<minimum_auc or mp!=1.:return False
    return True


def public_validation_order(episodes,results,source_revisions):
    """Finish public coverage before spending another round on known cases."""
    episodes=list(episodes)
    def pending(episode):
        result=results.get(str(episode))
        return not result or result.get('source_revision','')!=source_revisions.get(str(episode),'')
    return [ep for ep in episodes if pending(ep)]+[ep for ep in episodes if not pending(ep)]


def public_reconstruction_coverage_complete(results,episodes,source_revisions):
    """Source-video experiments permit further audits, never policy training."""
    if not episodes or set(results)!={str(ep) for ep in episodes}:return False
    for ep in episodes:
        result=results[str(ep)];baseline=result.get('baseline',{})
        if result.get('source_revision','')!=source_revisions.get(str(ep),''):return False
        if (baseline.get('kind')!='video_reconstructed_reference_not_policy' or
                baseline.get('episode')!=ep or baseline.get('evaluation_ground_truth_used') is not False or
                not isinstance(baseline.get('frames_scored'),int) or baseline['frames_scored']<=0):return False
        report=read(Path(result.get('overlap_report','')), {})
        if (report.get('episode')!=ep or report.get('all_source_frames_checked') is not True or
                not isinstance(report.get('object_frames_checked'),int) or report['object_frames_checked']<=0 or
                report.get('source_time_warping_used') is not False or
                report.get('physical_projection_applied') is not False):return False
    return True


def evaluation_source_coverage_pending(out,focus,policy):
    cad=read(Path(out)/'source-cad-status.json',{})
    version=('fresh_stereo_sift_frontend_v4' if policy.get('camera_odometry_method')=='sift_pnp_metric'
             else 'depth_markers_source_clock_v2')
    return cad.get('pipeline_version')!=version or any(str(ep) not in cad.get('results',{}) for ep in focus)


def cad_repair_input_changes(previous,episode):
    """Allow a repaired algorithm/review to run after an old retry budget.

    Compare the newest attempted CAD, including rejected candidates, rather
    than the retained reference. Unchanged exhausted attempts stay stopped.
    This does not alter reference selection or any source/physics criteria.
    """
    latest=latest_cad_attempt(previous)
    if not latest.get('refined_run'):return []
    research=Path(latest['refined_run'])/'research'
    report=read(research/'cad-alignment-report.json',{})
    changes=[]
    for name,old in report.get('algorithm_files_sha256',{}).items():
        path=BASE/name
        if path.is_file() and digest(path)!=old:changes.append(name)
    review=OUT/f'source-marker-overrides/public_{episode:06d}.json'
    binding=read(research/'source-marker-review.json',{}).get('binding') or {}
    current=digest(review) if review.is_file() else None
    if current!=binding.get('sha256'):
        changes.append('source-marker-review')
    return changes


def setup():
    ROOT.mkdir(exist_ok=True)
    for name in ['reconstruction','research','evaluation','videos','logs','scripts']:(ROOT/name).mkdir(exist_ok=True)
    prior=BASE/'runs/episode_000005_newton_20261015'
    for name,target in [('.venv-video',prior/'.venv-video'),('models',prior/'models')]:
        if not (ROOT/name).exists():(ROOT/name).symlink_to(target)
    # Read only roster/frame counts here. No public object pose values enter the
    # reconstruction process or its initialisation.
    import pyarrow.parquet as pq
    table=pq.read_table(BASE/f'datasets/video_to_data_challenge/track_3/public/data/chunk-000/episode_{EPISODE:06d}.parquet')
    names=json.loads(table.schema.metadata[b'objects'])
    cfg=dict(episode=EPISODE,sequence=f'episode_{EPISODE:06d}',run=str(ROOT),frames=len(table),objects=names,
             source=str(BASE/f'datasets/video_to_data_challenge/track_3/public/videos/chunk-000/observation.images.ego_cam_a/episode_{EPISODE:06d}.mp4'),
             fps=20.,resolution=[2028,1520],frame_cache_format='png',grasps=[],support_intervals={},
             evaluation_ground_truth_used=False,public_ground_truth_used_for_tracking=False,
             source_only_reconstruction=True,validated_video_pipeline=True,submission_mesh_face_budget=3600,
             photometric_refinement=True,
             source_feature_pose_enabled=True,
             static_joint_refinement=True,
             full_sequence_stereo_required=True,
             camera_odometry_method=read(OUT/'validated-pipeline-policy.json',{}).get('camera_odometry_method','droid_slam'),
             frontend_revision=SOURCE_REVISION,
             contour_pose_refinement_enabled=read(OUT/'validated-pipeline-policy.json',{}).get('contour_pose_refinement_enabled',False),
             planar_face_flip_proposals_enabled=read(OUT/'validated-pipeline-policy.json',{}).get('planar_face_flip_proposals_enabled',False),
             dataset_split='public',purpose='Video-only held-out public reconstruction validation')
    save(ROOT/'episode-config.json',cfg)
    previous=BASE/f'runs/public_episode_{EPISODE:06d}_pipeline_20261029'
    newer=previous.with_name(previous.name+'_v3')
    if '_pipeline_20261029_v4' in ROOT.name and (newer/'reconstruction/video-preparation.json').exists():previous=newer
    base_v4=BASE/f'runs/public_episode_{EPISODE:06d}_pipeline_20261029_v4'
    if SOURCE_REVISION and (base_v4/'reconstruction/video-preparation.json').exists():previous=base_v4
    if ROOT!=previous and (previous/'reconstruction/video-preparation.json').exists():
        # Camera images, hand detections and monocular depth are immutable
        # source measurements. Never copy old object poses or policies.
        for name in ['frames','calibrated.mp4','hand_landmarks_preview.mp4','hand_measurements.npz','camera_intrinsics.json','video-preparation.json',
                     'camera-frame-calibration.npz','camera-frame-calibration.json','sparse_stereo_geometry.npz',
                     'depth_foundation_stereo','foundation-stereo-sequence.json']:
            source=previous/'reconstruction'/name;target=ROOT/'reconstruction'/name
            if target.is_symlink():target.unlink()
            if source.is_dir():
                # Also complete a directory interrupted during a previous copy.
                shutil.copytree(source,target,copy_function=copy_measurement,dirs_exist_ok=True)
            elif source.is_file() and not target.exists():
                if source.suffix=='.json':shutil.copy2(source,target)
                else:copy_measurement(source,target)
        for name in ['sam2_prompts.json','source-prompt-review.json']:
            source=previous/'reconstruction'/name;target=ROOT/'reconstruction'/name
            if source.exists() and not target.exists():shutil.copy2(source,target)
        override=OUT/f'source-prompt-overrides/public_{EPISODE:06d}.json'
        if SOURCE_REVISION and override.exists() and not (ROOT/'reconstruction/masks-reviewed.json').exists():
            shutil.copy2(override,ROOT/'reconstruction/source-prompt-review.json')
        if SOURCE_REVISION and not (ROOT/'reconstruction/masks/1/000000.png').exists() and (ROOT/'reconstruction/source-prompt-review.json').exists():
            shutil.copy2(ROOT/'reconstruction/source-prompt-review.json',ROOT/'reconstruction/sam2_prompts.json')
        monocular=previous/'reconstruction/depth_moge'
        if (monocular/f"{cfg['frames']-1:06d}.png").exists() and not (ROOT/'reconstruction/depth').exists() and not (ROOT/'reconstruction/depth_moge').exists():
            (ROOT/'reconstruction/depth').symlink_to(monocular)
        save(ROOT/'research/source-measurement-reuse.json',dict(source=str(previous),images_hands_factory_and_stereo_calibration_reused=True,
             old_object_poses_reused=False,old_policy_reused=False,segmentation_rebuilt=True,ground_truth_used=False))
    if '_pipeline_20261029_v4' in ROOT.name and EPISODE==26:
        # Reviewed on the calibrated source RGB, independently of Vicon. The
        # previous detector box included the wooden tabletop in the pot mask.
        review=dict(prompts=[
            dict(object_id=1,frame_index=0,points=[dict(x=435.,y=454.),dict(x=488.,y=465.),dict(x=554.,y=586.),dict(x=585.,y=405.)],point_labels=[1,1,0,0]),
            dict(object_id=2,frame_index=0,points=[dict(x=581.,y=411.),dict(x=691.,y=502.),dict(x=554.,y=586.),dict(x=435.,y=454.)],point_labels=[1,1,0,0])],
            provenance='Manual image-only foreground/background points on public 26 frame zero; pot and rack separated from visible wood. No GT poses consulted.')
        for name in ['source-prompt-review.json','sam2_prompts.json']:
            target=ROOT/'reconstruction'/name
            if not target.exists() or read(target,{}).get('provenance')!=review['provenance']:save(target,review)
    return cfg


def source_prompts():
    """Image-only detections; keep every box and its source frame auditable."""
    import batch_worker as vision
    cfg=setup();destination=ROOT/'reconstruction/sam2_prompts.json'
    if destination.exists():return
    frames=sorted(set([0,cfg['frames']//4,cfg['frames']//2]))
    script=ROOT/'scripts/detect_source_objects.py'
    descriptions={'white_pot':'white saucepan with long handle','white_pot_lid':'white saucepan lid',
                  'mini_sweeper':'small broom brush','mini_dust_pan':'dustpan','blue_cup':'blue drinking cup',
                  'beige_cup':'beige drinking cup','water_pitcher':'water pitcher jug',
                  'plastic_dish_rack':'plastic dish drying rack','wooden_piece_1':'wooden board',
                  'wooden_piece_2':'wooden board','wooden_spoon':'wooden cooking spoon'}
    queries=[descriptions[name] for name in cfg['objects']]
    script.write_text('''from pathlib import Path\nfrom v2d.grounding_dino.lib.image_to_object_bboxes import image_to_bboxes\nROOT=Path('/episode')\nfor body,prompt in enumerate('''+repr(queries)+'''):
 for frame in '''+repr(frames)+''':
    image_to_bboxes(str(ROOT/f'reconstruction/frames/{frame:06d}.png'),str(ROOT/f'reconstruction/detections-{body}-{frame:06d}.json'),prompt+'.','/weights',box_threshold=.22,text_threshold=.2)
''')
    vision.docker(['run','--rm','--gpus','all','--ipc','host','-v',f'{ROOT}:/episode','-v',f'{vision.MODULES}:/workspace','-v',f'{vision.WEIGHTS}/grounding_dino:/weights:ro','v2d_grounding_dino','python','/episode/scripts/detect_source_objects.py'],ROOT/'logs/grounding-dino.log')
    if cfg['objects']==['wooden_piece_1','wooden_piece_2']:
        # Identical text queries must not initialise both IDs on one board.
        # Spatial ordering only initialises the tracker; marker identity still
        # needs independent reconstruction/validation, and is not asserted here.
        for frame in frames:
            detections=read(ROOT/f'reconstruction/detections-0-{frame:06d}.json',[])
            chosen=distinct_board_pair(detections)
            if chosen:
                prompts=[dict(object_id=i+1,frame_index=frame,box=row['box']) for i,row in enumerate(chosen)]
                save(destination,dict(prompts=prompts,provenance='Distinct source-image wooden-board detections, initially ordered left to right; no pose GT used.',object_marker_identity_verified=False))
                return
        raise RuntimeError('Two distinct wooden boards were not detected; refusing overlapping or nested object initialisation boxes')
    prompts=[]
    for body,name in enumerate(cfg['objects']):
        choices=[]
        for frame in frames:
            for detection in read(ROOT/f'reconstruction/detections-{body}-{frame:06d}.json',[]):
                choices.append((float(detection['confidence']),frame,detection))
        if not choices:raise RuntimeError(f'Grounding DINO found no unambiguous {name}; source-image prompt review needed')
        confidence,frame,detection=max(choices,key=lambda row:row[0])
        prompts.append(dict(object_id=body+1,frame_index=frame,box=detection['box']))
    validate_distinct_detection_prompts(prompts)
    save(destination,dict(prompts=prompts,provenance='Grounding DINO on calibrated public RGB frames only; no Vicon poses, masks or geometry used for detections.',detections_frames=frames))


def masks_match_prompts(root):
    rec=Path(root)/'reconstruction'
    cfg=read(Path(root)/'episode-config.json',{})
    prompt=rec/'sam2_prompts.json'
    report=read(rec/'masks/propagation-report.json',{})
    return (prompt.exists() and report.get('prompts_sha256')==digest(prompt)
            and all((rec/f"masks/{body}/{cfg['frames']-1:06d}.png").exists() for body in range(1,len(cfg['objects'])+1)))


def reviewed_masks():
    import batch_worker as vision
    from source_cap_masks import refine as refine_source_caps
    from source_background_review import apply_review
    def refine_masks():
        refine_source_caps(ROOT)
        apply_review(ROOT)
    def rerun_masks(command):
        vision.docker(command,ROOT/'logs/sam2-auto-repair.log')
        refine_masks()
    review=ROOT/'reconstruction/source-prompt-review.json'
    if review.exists():
        original=ROOT/'reconstruction/sam2_prompts.dino.json'
        if not original.exists():shutil.copy2(ROOT/'reconstruction/sam2_prompts.json',original)
        changed=digest(review)!=digest(ROOT/'reconstruction/sam2_prompts.json')
        shutil.copy2(review,ROOT/'reconstruction/sam2_prompts.json')
        command=read(BASE/'runs/episode_000005_newton_20261015/research/sam2-command.json')[1:]
        command=[x.replace(str(BASE/'runs/episode_000005_newton_20261015'),str(ROOT)) for x in command]
        if changed or not masks_match_prompts(ROOT):
            vision.docker(command,ROOT/'logs/sam2-reviewed.log')
    refine_masks()
    if read(ROOT/'episode-config.json').get('camera_odometry_method')=='sift_pnp_metric':
        from source_mask_quality import inspect_and_repair,audit as audit_masks
        command=read(BASE/'runs/episode_000005_newton_20261015/research/sam2-command.json')[1:]
        command=[x.replace(str(BASE/'runs/episode_000005_newton_20261015'),str(ROOT)) for x in command]
        if read(ROOT/'episode-config.json').get('fast_submission'):
            report=audit_masks(ROOT)
            save(ROOT/'research/source-mask-quality.json',dict(latest=report,attempts=[dict(attempt=0,**report)],
                quality_is_diagnostic=True,strict_completion_granted=False))
        else:inspect_and_repair(ROOT,lambda:rerun_masks(command))
    alignment=ROOT/'reconstruction/stereo-depth-alignment.json'
    cap_masks=read(ROOT/'research/source-cap-mask-refinement.json',{}).get('masks_sha256')
    background=ROOT/'reconstruction/source-background-review.json'
    background_sha=digest(background) if background.exists() else None
    depth_inputs=read(alignment,{})
    if alignment.exists() and (depth_inputs.get('source_masks_prompt_sha256')!=digest(ROOT/'reconstruction/sam2_prompts.json')
                             or depth_inputs.get('source_cap_masks_sha256')!=cap_masks
                             or depth_inputs.get('source_background_review_sha256')!=background_sha):
        # Mask changes also change the metric scale fit used to fill stereo
        # holes with MoGE. Refresh this CPU stage before fitting object poses.
        vision.run([str(PY),str(ROOT/'scripts/dense_stereo_depth.py')],ROOT/'logs/depth-after-mask-review.log')
        depth_report=read(alignment,{})
        depth_report.update(source_masks_prompt_sha256=digest(ROOT/'reconstruction/sam2_prompts.json'),
                            source_cap_masks_sha256=cap_masks,
                            source_background_review_sha256=background_sha,
                            mask_review_depth_refreshed_utc=utc(),dataset_split=read(ROOT/'episode-config.json').get('dataset_split','evaluation'))
        save(alignment,depth_report)
    save(ROOT/'reconstruction/masks-reviewed.json',dict(updated_utc=utc(),source_prompt_review=review.exists(),
         source_prompt_sha256=digest(ROOT/'reconstruction/sam2_prompts.json'),ground_truth_used=False,
         exact_mask_accuracy_verified=False))


def run_vision(stage):
    import batch_worker as vision
    cfg=setup();original=vision.run
    def public_run(command,log):
        # Generated template scripts originally point at evaluation videos.
        # Remap video paths only; calibration remains public/meta for all splits.
        if len(command)>1:
            target=Path(str(command[1]))
            if target.suffix=='.py' and target.is_file() and target.is_relative_to(ROOT):
                text=target.read_text().replace('evaluation/videos/','public/videos/').replace('frames of evaluation ego','frames of public ego')
                target.write_text(text)
        return original(command,log)
    vision.run=public_run
    if stage=='prepare':vision.prepare(ROOT,cfg)
    elif stage=='prompts':source_prompts()
    elif stage=='masks':reviewed_masks()
    elif stage=='metric':
        vision.metric(ROOT,cfg)
        for name in ['camera-frame-calibration.json','stereo-depth-alignment.json']:
            path=ROOT/'reconstruction'/name;report=read(path,{})
            report.update(dataset_split='public',public_ground_truth_used_for_tracking=False)
            if isinstance(report.get('source'),str):report['source']=report['source'].replace('frames of evaluation ego','frames of public ego')
            save(path,report)
    elif stage=='generic':vision.generic_reconstruction(ROOT,cfg)
    elif stage=='stereo':
        from foundation_stereo_sequence import run
        run(ROOT)
    elif stage=='odometry':
        from sift_metric_odometry import run
        run(ROOT)
    else:raise ValueError(stage)


def worker(out,focus):
    VAL.mkdir(exist_ok=True);state=read(out/'status.json',{})
    policy=read(out/'validated-pipeline-policy.json',{})
    plan=state.setdefault('execution_plan',{})
    for ep in focus:
        plan.setdefault(str(ep),{}).update(episode=ep,automatic=True,status='queued_public_validation',
            queue_position=focus.index(ep)+1,episode_done=False,
            stages=['fresh_video_frontend','CAD_source_alignment','physical_reference','policy_training','official_rollout','strict_source_and_contact_checks'])
    state.update(execution_plan=plan,focused_episodes=focus,expected_episodes=focus,serial_queue_order=focus,parked_episodes=[])
    save(out/'execution-plan.json',dict(updated_utc=utc(),automatic=True,episodes=plan))
    status=read(VAL/'status.json',{})
    failure={key:status.pop(key,None) for key in ['error','error_report']}
    if any(failure.values()):status['previous_validation_failure']=dict(**failure,archived_utc=utc())
    status.update(status='running',active_episode=0,dataset_split='public',
          evaluation_scope=focus,pid=os.getpid(),run=str(ROOT),training_steps=0,policy_training_started=False)
    def emit(phase):
        status.update(updated_utc=utc(),phase=phase);save(VAL/'status.json',status)
        state.update(updated_utc=utc(),active=True,status='public_pipeline_validation',current_episode=EPISODE,
                     active_optimization={'episode':EPISODE,'run':str(ROOT),'pid':os.getpid(),'action':phase,'dataset_split':'public'},
                     public_validation=str(VAL/'status.json'),episodes_done=[],pending_episodes=focus)
        save(out/'status.json',state)
    def command(args,label):
        emit(label)
        with (ROOT/f'logs/{label}.log').open('a') as log:
            child=subprocess.Popen([str(x) for x in args],cwd=BASE,stdout=log,stderr=subprocess.STDOUT)
            register_episode_child(child.pid)
            while child.poll() is None:emit(label);time.sleep(10)
            if child.returncode:raise RuntimeError(f'{label} failed ({child.returncode}); see {ROOT}/logs/{label}.log')
    def yield_for_requested_reload():
        if not read(OUT/'controller-reload-request.json',{}).get('pending'):return False
        status.update(status='awaiting_safe_controller_reload',active_episode=None,evaluation_training_admitted=False)
        emit('Completed protected episode; waiting for the existing health monitor to reload the serial controller')
        return True
    try:
        results=status.setdefault('results',{})
        def yield_for_evaluation_audits():
            if not policy.get('evaluation_reconstruction_after_public_coverage'):return False
            revisions=read(OUT/'frontend-revisions.json',{})
            episodes=policy.get('public_episodes',[0,2,18,21,23,26,31])
            if (not evaluation_source_coverage_pending(out,focus,policy) or
                    references_passed(results,episodes,policy.get('minimum_public_auc',.9),revisions) or
                    not public_reconstruction_coverage_complete(results,episodes,revisions)):return False
            status.update(status='public_reference_validation_quality_insufficient',active_episode=None,
                          evaluation_training_admitted=False,evaluation_reconstruction_admitted=True)
            emit('Public source-video experiments covered; audit all evaluation videos next; policy training stays gated')
            return True
        if yield_for_evaluation_audits():return False
        order=public_validation_order(policy.get('public_episodes',[0,2,18,21,23,26,31]),
                                      results,read(OUT/'frontend-revisions.json',{}))
        status['public_validation_order']=order
        for ep in order:
            configure(ep)
            status.update(active_episode=ep,run=str(ROOT));status.pop('cad_run',None)
            setup();key=str(ep)
            previous=results.get(key,{})
            if previous and previous.get('source_revision','')!=SOURCE_REVISION:
                save(VAL/f'superseded-public-{ep:06d}-{time.time_ns()}.json',previous)
                results.pop(key,None);previous={}
            repair=bool(previous)
            retry_changes=[]
            if previous:
                metrics=previous.get('best_reference_metrics',previous['refined_metrics'])
                if metrics['AUC']>=policy.get('minimum_public_auc',.9) and metrics['MP-SR']==1.:continue
                if previous.get('repair_round',0)>=3:
                    retry_changes=cad_repair_input_changes(previous,ep)
                    if not retry_changes:continue
            repair_round=previous.get('repair_round',0)+1 if repair else 0
            status.update(active_episode=ep,run=str(ROOT))
            status.pop('cad_run',None)
            with episode_slot(ROOT,ep):
                stages=[('prepare','video-preparation.json'),('prompts','sam2_prompts.json'),('metric','stereo-depth-alignment.json'),('masks','masks-reviewed.json')]
                if policy.get('camera_odometry_method')=='sift_pnp_metric':
                    stages += [('stereo','foundation-stereo-sequence.json'),('odometry','sift-pnp-odometry.json')]
                stages += [('generic','scene_motion_physical.npz')]
                for stage,marker in stages:
                    if not (ROOT/'reconstruction'/marker).exists():
                        command([PY,BASE/'public_validation_pipeline.py','--episode',ep,'--stage',stage],f'public-{stage}')
                if repair:
                    command([PY,BASE/'foundation_stereo_sequence.py','--run',ROOT],'public-foundation-stereo')
                pred=ROOT/'reconstruction/scene_motion_physical.npz';meshes=ROOT/'reconstruction/mesh_priors'
                baseline=VAL/f"public_{ep:06d}{policy.get('score_revision','')}{SOURCE_REVISION}_reference_baseline.json"
                if not baseline.exists():command([PY,BASE/'t3_local_score.py','--public-episode',ep,'--prediction',pred,'--mesh-dir',meshes,'--output',baseline],'public-score-baseline')
                suffix=policy.get('score_revision','')+SOURCE_REVISION+(f'_repair{repair_round}' if repair else '')
                version=policy.get('cad_alignment_run_suffix','')
                aligned=BASE/f'runs/public_episode_{ep:06d}_cad_alignment_20261029{version}{suffix}'
                status['cad_run']=str(aligned)
                report=aligned/'research/cad-alignment-report.json'
                if not report.exists():command([PY,BASE/'source_cad_alignment.py','--source',ROOT,'--run',aligned,'--maxfev',100*(repair_round+1)],'public-cad-refinement')
                scores={}
                for advance in [-2.,-1.25,0.,1.25,2.]:
                    output=VAL/f'public_{ep:06d}{suffix}_advance_{advance:+.2f}.json'
                    if not output.exists():command([PY,BASE/'t3_local_score.py','--public-episode',ep,'--prediction',aligned/'reconstruction/cad-source-trajectory.npz','--mesh-dir',aligned/'reconstruction/mesh_priors','--advance',advance,'--skip-mesh','--output',output],f'public-offset-{advance:+.2f}')
                    scores[str(advance)]=read(output)['metrics']
                # Selection evidence is kept apart from the source-only fitter.
                candidate=dict(baseline=read(baseline),refined_metrics=scores['0.0'],offset_sweep=scores,
                               overlap_report=str(report),refined_run=str(aligned),repair_round=repair_round,source_revision=SOURCE_REVISION,
                               repaired_inputs_since_exhausted_attempt=retry_changes)
                base=candidate['baseline']['metrics'];new=candidate['refined_metrics']
                candidate['adopted_over_baseline']=all(new[k]>=base[k]-1e-9 for k in ['AUC','SP-SR','MP-SR']) and all(new[k]<=base[k]+1e-9 for k in ['MPPE_cm','RPE_cm'])
                candidate['best_reference_metrics']=new if candidate['adopted_over_baseline'] else base
                selected=aligned/'reconstruction/cad-source-trajectory.npz' if candidate['adopted_over_baseline'] else pred
                candidate['selected_reference_prediction']=str(selected)
                candidate['selected_reference_sha256']=digest(selected)
                candidate['selection_reason']='All measured public pose metrics maintained or improved' if candidate['adopted_over_baseline'] else 'Reference regression rejected; preserve the source baseline while testing another reconstruction'
                results[key]=select_attempt(previous,candidate)
                save(VAL/'status.json',status)
            if yield_for_requested_reload():return False
            if yield_for_evaluation_audits():return False
        if yield_for_requested_reload():return False
        passed=references_passed(results,policy.get('public_episodes',[0,2,18,21,23,26,31]),policy.get('minimum_public_auc',.9),read(OUT/'frontend-revisions.json',{}))
        status.update(status='public_reference_validation_passed' if passed else 'public_reference_validation_quality_insufficient',
                      active_episode=None,evaluation_time_shift_authorized=False,evaluation_training_admitted=passed,
                      public_identity_control_is_not_reconstruction_quality=True,
                      reason='Frame indices preserved. The proposed 1.25-frame offset is diagnostic only; no global shift adopted. Public scans do not include the wooden spoon.')
        emit('Public reconstruction validation passed; evaluation CAD audit next' if passed else 'Public reference quality remains below admission criteria; evaluation training stays gated')
        return passed
    except Exception:
        error=traceback.format_exc();error_path=ROOT/'logs/public-validation-exception.txt'
        error_path.parent.mkdir(parents=True,exist_ok=True);error_path.write_text(error)
        status.update(status='failed',error=error[-4000:],error_report=str(error_path));emit('Public validation failed; evaluation training remains blocked');return False


if __name__=='__main__':
    import argparse
    parser=argparse.ArgumentParser();parser.add_argument('--stage',choices=['prepare','prompts','metric','masks','stereo','odometry','generic'],required=True);parser.add_argument('--episode',type=int,default=0)
    args=parser.parse_args();configure(args.episode);run_vision(args.stage)
