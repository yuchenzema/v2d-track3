"""Run the unchanged submission-kit evaluator; retain genuine trajectory provenance.

Reference/self scores are diagnostic, never Kaggle held-out scores. No object
trajectory is generated from the reconstruction NPZ or copied from a target.
"""
import argparse
import hashlib
import json
import shutil
from pathlib import Path
import numpy as np
import pyarrow.parquet as pq

from improvement_common import BASE,save,utc,digest
from batch_native import _docker,docker

KIT=BASE/'v2d_submission_kit'
POSE_COLUMNS=['achieved_'+x for x in ['position_x','position_y','position_z','quaternion_x','quaternion_y','quaternion_z','quaternion_w']]


def verify_trajectory(path,checkpoint_sha,eval_script_sha,worlds,episode):
    table=pq.read_table(path);meta=table.schema.metadata
    expected={b'flash_chord.checkpoint_sha256':checkpoint_sha.encode(),
              b'flash_chord.eval_script_sha256':eval_script_sha.encode(),
              b'flash_chord.world_count':str(worlds).encode(),
              b'flash_chord.start_frame':b'0',b'flash_chord.quaternion_order':b'xyzw',
              b'flash_chord.episode_index':str(episode).encode()}
    for name,value in expected.items():assert meta.get(name)==value,('Unverified official rollout provenance',name)
    assert meta.get(b'flash_chord.eval_code_sha256'),'Missing evaluation runtime hash'
    values=np.column_stack([table[name].to_numpy() for name in POSE_COLUMNS])
    assert np.isfinite(values).all() and table['environment_valid'].to_numpy().all(),'Official rollout contains a failed/non-finite world'
    assert np.allclose(np.linalg.norm(values[:,3:],axis=1),1.,atol=1e-4),'Invalid rollout quaternion'
    return {k.decode():v.decode() for k,v in meta.items() if k.startswith(b'flash_chord.')}


def evaluate_candidate(root,checkpoint,container):
    root=Path(root);cfg=json.loads((root/'episode-config.json').read_text());ep=cfg['episode']
    local=root/str(checkpoint).removeprefix('/episode/') if str(checkpoint).startswith('/episode/') else Path(checkpoint)
    local=local.resolve();relative=local.relative_to(root.resolve());checkpoint_sha=digest(local)
    script=KIT/'flash_chord/scripts/evaluate_policy.py';script_sha=digest(script)
    kit_files={str(p.relative_to(KIT)):digest(p) for p in KIT.rglob('*') if p.is_file() and '__pycache__' not in p.parts}
    kit_hash=hashlib.sha256(json.dumps(kit_files,sort_keys=True).encode()).hexdigest()
    container_kit=f'/tmp/v2d-official-kit-{kit_hash[:16]}'
    folder=root/f'evaluation/official/{checkpoint_sha[:16]}';folder.mkdir(parents=True,exist_ok=True)
    status=dict(status='running',updated_utc=utc(),episode=ep,checkpoint=str(local),checkpoint_sha256=checkpoint_sha,
                official_kit_files_sha256=kit_files,official_entrypoint_sha256=script_sha,
                kind='actual_unassisted_policy_rollouts',reference_scores_are_not_kaggle_scores=True,attempts={})
    old=folder/'suite.json'
    if old.exists():status['attempts']=json.loads(old.read_text()).get('attempts',{})
    save(old,status)
    input_adapter=[]
    if cfg.get('validated_video_pipeline'):
        helpers=['source_reference_entrypoint.py','legacy_source_frame_playback.py','official_body_recording.py']
        for helper in helpers:shutil.copy2(BASE/helper,root/'scripts'/helper)
        input_adapter=['/episode/scripts/source_reference_entrypoint.py']
        status['reference_input_adapter_sha256']={name:digest(BASE/name) for name in helpers}
        status['reference_input_adapter_scope']='Legacy HandPose source-frame loading only; no pose edits, time interpolation or evaluator/metric code changes'
    _docker(['cp',str(KIT),f'{container}:{container_kit}'],root/'logs/official-kit-copy.log')
    # Verify every copied file before starting the official executable. This
    # checks evaluator bytes only. A separately hashed reference input adapter
    # is declared above for the legacy HandPose schema used by these episodes.
    verify='import hashlib,json;from pathlib import Path;d=json.loads('+repr(json.dumps(kit_files))+');r=Path('+repr(container_kit)+');assert all(hashlib.sha256((r/p).read_bytes()).hexdigest()==h for p,h in d.items())'
    _docker(['exec',container,'python','-c',verify],root/'logs/official-kit-verification.log')
    labels=([('recorded',1)] if cfg.get('official_recorded_only') else
            [('self16',16)]+[(f'repeat{i:02d}',1) for i in range(1,7)]+[('recorded',1)])
    for label,worlds in labels:
        result=folder/label;result.mkdir(exist_ok=True)
        parquet=result/f'episode_{ep:06d}.parquet';report=result/f'episode_{ep:06d}.json'
        body_states=result/'official-body-states.npz'
        if not (parquet.exists() and report.exists() and (label!='recorded' or body_states.exists())):
            target='/episode/'+str(result.relative_to(root))
            recording_env=[]
            if label=='recorded':
                recording_env=[f'V2D_OFFICIAL_BODY_STATES={target}/official-body-states.npz',f'V2D_OFFICIAL_CHECKPOINT_SHA256={checkpoint_sha}']
                if parquet.exists() and report.exists():
                    history=result/'history'/digest(parquet)[:16];history.mkdir(parents=True,exist_ok=True)
                    for prior in [parquet,report]:shutil.copy2(prior,history/prior.name)
            command=['exec','--workdir','/workspace',container,'env','XLA_PYTHON_CLIENT_PREALLOCATE=false',
                     *recording_env,
                     'python','-u',*input_adapter,container_kit+'/flash_chord/scripts/evaluate_policy.py',
                     'evaluation.checkpoint=/episode/'+str(relative),f'evaluation.world_count={worlds}',
                     'evaluation.reset_mode=explicit','evaluation.start_frame=0','logging.mode=disabled',
                     f'evaluation.metrics_output={target}/episode_{ep:06d}.json',
                     f'evaluation.object_trajectories_output={target}/episode_{ep:06d}.parquet',
                     f'hydra.run.dir={target}/hydra']
            docker(command,root/f'logs/official-{checkpoint_sha[:12]}-{label}.log')
        data=json.loads(report.read_text())
        assert data['reset_mode']=='explicit' and data['start_frame']==0 and data['world_count']==worlds
        provenance=verify_trajectory(parquet,checkpoint_sha,script_sha,worlds,ep)
        if cfg.get('validated_video_pipeline'):
            assert int(provenance['flash_chord.step_count'])==cfg['frames'],'Official evaluator clock differs from the source video frame grid'
        status['attempts'][label]=dict(world_count=worlds,metrics=data['metrics'],
             actual_training_steps=data['environment_steps'],trajectory=str(parquet),trajectory_sha256=digest(parquet),provenance=provenance)
        status.update(updated_utc=utc(),active_attempt=label);save(old,status)
    success=all(v['metrics']['chord_sr']==1. for v in status['attempts'].values())
    status.update(status='official_evaluation_complete',updated_utc=utc(),active_attempt=None,
                  all_reference_trials_succeeded=success,recorded_trajectory=status['attempts']['recorded']['trajectory'],
                  recorded_body_states=str(folder/'recorded/official-body-states.npz'),
                  strict_source_motion_verified=False,contact_validation_still_required=True,
                  manual_review_still_required=True,submission_ready=False)
    if cfg.get('post_training_grasp_acceptance_required'):
        from post_training_grasp_acceptance import evaluate
        status['post_training_grasp_acceptance']=evaluate(root,status)
    save(old,status);return status


def select_recorded_attempt(root):
    """Render official states and recompute diagnostics for that exact attempt."""
    root=Path(root);cfg=json.loads((root/'episode-config.json').read_text())
    path=root/'final-controller-selection.json';selection=json.loads(path.read_text());selected=selection['selected']
    checkpoint=root/str(selected['checkpoint']).removeprefix('/episode/')
    if Path(selected['checkpoint']).is_absolute() and not str(selected['checkpoint']).startswith('/episode/'):checkpoint=Path(selected['checkpoint'])
    sha=digest(checkpoint);suite=json.loads((root/f'evaluation/official/{sha[:16]}/suite.json').read_text())
    states=Path(suite['recorded_body_states'])
    with np.load(states,allow_pickle=False) as record:
        assert bool(record['policy_inference']) and str(record['checkpoint_sha256'])==sha
        assert len(record['body_poses'])==cfg['frames'] and np.all(record['object_assistance']==0.)
    state_sha=digest(states)
    if (selected.get('official_recorded_state_sha256')==state_sha
            and selected.get('source_fidelity_report') and (root/selected['source_fidelity_report']).is_file()):
        bind_rendered_attempt(root)
        return selection
    save(root/f'research/pre-official-selection-{digest(path)[:16]}.json',selection)
    from batch_review import cpu
    label=f'official-selected-{state_sha[:12]}'
    cpu(root,[str(root/'scripts/assess_source_fidelity.py'),'--states',str(states),'--output',label+'-source'],label+'-source')
    tracking=root/f'evaluation/{label}-tracking.json'
    cpu(root,[str(root/'scripts/assess_single_attempt.py'),'--states',str(states),'--reference',cfg['training_reference'],'--output',str(tracking)],label+'-tracking')
    fidelity=json.loads((root/f'evaluation/{label}-source.json').read_text())
    selected.update(recorded_states=str(states.relative_to(root)),single_attempt_tracking=json.loads(tracking.read_text()),
        source_mean_object_iou=sum(v['iou'] for v in fidelity['aggregate']['objects'].values())/len(cfg['objects']),
        source_worst_phase_object_iou=min(v['mean_iou'] for phase in fidelity['phases'] for v in phase['objects'].values()),
        official_recorded_state_sha256=state_sha,
        source_fidelity_report=f'evaluation/{label}-source.json',
        source_tracking_report=str(tracking.relative_to(root)),
        recording_protocol=dict(simulation_world_count=1,recorded_world_index=0,
            world_selection='Official predetermined single-world recorded attempt; no success-based selection'))
    selected['source_agreement_score']=selected['source_mean_object_iou']+.25*selected['source_worst_phase_object_iou']
    from rollout_collision_validation import audit_rollout
    selected['collision_validation']=audit_rollout(root,states,label,cfg['training_reference'])
    save(path,selection)
    bind_rendered_attempt(root)
    return selection


def quaternion_discrepancy(first,second):
    first=np.asarray(first,dtype=np.float64);second=np.asarray(second,dtype=np.float64)
    first=first/np.linalg.norm(first,axis=1,keepdims=True)
    second=second/np.linalg.norm(second,axis=1,keepdims=True)
    return float(np.max(2*np.arccos(np.clip(np.abs(np.sum(first*second,axis=1)),0.,1.))))


def bind_rendered_attempt(root):
    """Require video object states to reproduce the actual official recording."""
    root=Path(root);selected=json.loads((root/'final-controller-selection.json').read_text())['selected']
    checkpoint=root/selected['checkpoint'].removeprefix('/episode/');sha=digest(checkpoint)
    suite=json.loads((root/f'evaluation/official/{sha[:16]}/suite.json').read_text())
    parquet=Path(suite['recorded_trajectory']);table=pq.read_table(parquet)
    path=Path(selected['recorded_states'])
    states=root/str(path).removeprefix('/episode/') if str(path).startswith('/episode/') else path if path.is_absolute() else root/path
    with np.load(states,allow_pickle=False) as recording:
        names=recording['body_names'].tolist();poses=recording['body_poses']
        assert np.all(recording['object_assistance']==0.)
        assert len(poses)==int(table.schema.metadata[b'flash_chord.step_count'])
        official=np.column_stack([table[name].to_numpy() for name in POSE_COLUMNS])
        frames=table['reference_frame'].to_numpy()
        ids=np.array([names.index(name if name in names else f'track3_{name}/object') for name in table['object_body_name'].to_pylist()])
        rendered=poses[frames,ids]
    translation=float(np.max(np.linalg.norm(rendered[:,:3]-official[:,:3],axis=1)))
    orientation=quaternion_discrepancy(rendered[:,3:],official[:,3:])
    result=dict(updated_utc=utc(),checkpoint_sha256=sha,official_trajectory=str(parquet),official_trajectory_sha256=digest(parquet),
                rendered_state_sha256=digest(states),maximum_object_translation_difference_m=translation,
                maximum_object_orientation_difference_rad=orientation,
                rendered_objects_match_official_recording=translation<=1e-4 and orientation<=1e-3,
                temporal_resampling_used=False,frame_splicing_used=False)
    save(root/'evaluation/official-render-binding.json',result)
    assert result['rendered_objects_match_official_recording'],'Rendered object trajectories differ from the official single-world recording; publication blocked'
    return result


if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('--run',type=Path,required=True);parser.add_argument('--checkpoint',required=True);parser.add_argument('--container',required=True)
    a=parser.parse_args();evaluate_candidate(a.run,a.checkpoint,a.container)
