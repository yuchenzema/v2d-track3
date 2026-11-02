"""Screen source-derived grasp flexion corrections without changing any learner."""
from pathlib import Path
import argparse
import hashlib
import itertools
import json
import shutil
from batch_native import docker
from improvement_common import BASE,BATCH,root,baseline,digest,save,utc,verify_preserved


def sweep(ep):
    original=baseline(ep);source=root(ep)/original['checkpoint'].removeprefix('/episode/')
    target=BASE/'runs'/f'episode_{ep:06d}_newton_grasp_curriculum_20261020'
    cfg=json.loads((target/'episode-config.json').read_text());sha=digest(source)
    checkpoint=target/'research'/f'baseline-for-grip-{sha[:16]}.safetensors'
    if not checkpoint.exists():shutil.copy2(source,checkpoint)
    assert digest(checkpoint)==sha
    for name in ['grip_bias_controller.py','evaluate_grip_bias_sweep.py']:
        shutil.copy2(BASE/name,target/'scripts'/name)
    profiles=[{'left':left,'right':right} for left,right in itertools.product([0.,.08,.16,.24],repeat=2)]
    code={name:digest(BASE/name) for name in ['grip_bias_controller.py','evaluate_grip_bias_sweep.py']}
    identity={'checkpoint_sha256':sha,'profiles':profiles,'controller_code':code,
        'physics_substeps':original['physics_substeps'],'reference':original['inference_reference'],
        'velocity_gain':original['wrist_reference_velocity_gain']}
    tag=hashlib.sha256(json.dumps(identity,sort_keys=True).encode()).hexdigest()[:12]
    diagnostic=target/f'evaluation/grip-bias-sweep-{tag}.json'
    provenance={'episode':ep,'updated_utc':utc(),'diagnostic_only':True,
        'method':'Different smooth finger flexion corrections in 16 physical worlds; unchanged learned baseline actor',
        'checkpoint_sha256':sha,'original_physics_substeps':original['physics_substeps'],
        'trial_identity':identity,
        'object_assistance':0.,'policy_training_interrupted':False,
        'requires_single_profile_validation_against_all_original_video_frames':True}
    marker=BATCH/'improvements_20261020_grasp'/f'episode_{ep:06d}-grip-sweep-status.json'
    save(marker,{**provenance,'status':'testing'})
    if not diagnostic.exists():
        docker(['exec',cfg['native_container'],'env',f'V2D_GRIP_BIAS_PROFILES={json.dumps(profiles)}',
            'V2D_GRIP_SWEEP_OUTPUT=/episode/'+str(diagnostic.relative_to(target)),
            f'V2D_WRIST_VELOCITY_GAIN={original["wrist_reference_velocity_gain"]}',
            f'V2D_SIM_SUBSTEPS={original["physics_substeps"]}',
            f"V2D_REFERENCE_OVERRIDE=/episode/human_motion_data/{original['inference_reference']}/processed/sequence_id={cfg['sequence']}/robot_name=sharpa_wave",
            'python','-u','/episode/scripts/evaluate_grip_bias_sweep.py',
            'evaluation.checkpoint=/episode/'+str(checkpoint.relative_to(target)),
            'evaluation.world_count=16','logging.mode=disabled',
            f'evaluation.metrics_output=/episode/evaluation/grip-bias-sweep-{tag}-aggregate-not-a-cohort-score.json',
            f'hydra.run.dir=/episode/hydra/grip-bias-sweep-{tag}'],target/f'logs/grip-bias-sweep-{tag}.log')
    result=json.loads(diagnostic.read_text())
    assert result['full_native_control_frames']==2*(cfg['frames']-1)
    assert digest(source)==sha and digest(checkpoint)==sha
    verify_preserved(ep)
    save(marker,{**provenance,'updated_utc':utc(),'status':'completed','diagnostic':str(diagnostic),'results':result})
    print(ep,[(v['profile'],v['first_violation_control_frame'],v['violation_frame_count']) for v in result['profiles']],flush=True)
    return result


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('episodes',type=int,nargs='+')
    args=parser.parse_args()
    for ep in args.episodes:sweep(ep)
