"""Test a preserved learned actor on another episode's unchanged full reference.

Trials are diagnostic until they pass the ordinary per-object source comparison,
larger cohort validation, rendering and artifact checks. They never replace an
active learner, its checkpoints, or the episode's best published policy.
"""
from pathlib import Path
import argparse
import json
import shutil
import traceback
from batch_review import review
from improvement_common import BASE,BATCH,baseline,better,digest,save,utc,verify_preserved


def trial(ep,source_episode=15):
    source=BASE/'runs'/f'episode_{source_episode:06d}_newton_improvement_20261019'
    selected=json.loads((source/'final-controller-selection.json').read_text())['selected']
    source_checkpoint=source/selected['checkpoint'].removeprefix('/episode/')
    expected=digest(source_checkpoint)
    if expected!=selected['checkpoint_sha256']:
        raise ValueError('The selected source actor changed')
    target=BASE/'runs'/f'episode_{ep:06d}_newton_grasp_curriculum_20261020'
    cfg=json.loads((target/'episode-config.json').read_text())
    directory=target/'research/cross_episode_transfer'
    directory.mkdir(exist_ok=True)
    checkpoint=directory/f'episode_{source_episode:06d}-{expected[:16]}.safetensors'
    if not checkpoint.exists():shutil.copy2(source_checkpoint,checkpoint)
    if digest(checkpoint)!=expected:raise ValueError('Copied actor checksum mismatch')
    result_path=BATCH/'improvements_20261020_grasp'/f'episode_{ep:06d}-transfer-trial.json'
    provenance={'episode':ep,'source_episode':source_episode,'updated_utc':utc(),
        'method':'Unmodified learned actor evaluated on the target full reference and geometry',
        'source_checkpoint':str(source_checkpoint),'source_checkpoint_sha256':expected,
        'target_run':str(target),'target_training_steps_added':0,
        'evaluation_object_assistance':0.,'published_as_improvement':False}
    save(result_path,{**provenance,'status':'testing'})
    try:
        row=review(ep,selected['environment_steps'],selected['wrist_reference_velocity_gain'],
            cfg['training_reference'],cfg['physics_substeps'],
            checkpoint_override='/episode/'+str(checkpoint.relative_to(target)),
            root_override=target,container_override=cfg['native_container'])
        if digest(source_checkpoint)!=expected or digest(checkpoint)!=expected:
            raise ValueError('Transfer trial modified the learned actor')
        verify_preserved(ep)
        result={**provenance,'updated_utc':utc(),'status':'completed','candidate':row,
            'passes_initial_improvement_comparison':better(row,baseline(ep)),
            'requires_128_world_validation_and_validated_videos_before_publication':True}
    except Exception as exc:
        result={**provenance,'updated_utc':utc(),'status':'failed',
            'error':str(exc),'traceback':traceback.format_exc()}
    save(result_path,result)
    print(json.dumps({k:v for k,v in result.items() if k not in ['candidate','traceback'] }),flush=True)
    return result


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('episodes',type=int,nargs='+')
    parser.add_argument('--source',type=int,default=15)
    args=parser.parse_args()
    for episode in args.episodes:trial(episode,args.source)
