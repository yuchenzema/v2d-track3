"""Check an unassisted opening grasp; never publish it as a full rollout."""
from pathlib import Path
import argparse
import json
import struct
from batch_native import docker
from improvement_common import BASE,BATCH,save,utc,digest


def evaluate(ep,step,backward_stage=None,variant='grasp_curriculum_20261020'):
    assert variant in ['grasp_curriculum_20261020','backward_grasp_20261020']
    root=BASE/'runs'/f'episode_{ep:06d}_newton_{variant}'
    cfg=json.loads((root/'episode-config.json').read_text())
    prefix=root/'grasp_pretraining'
    if backward_stage is not None:
        assert 0<=backward_stage<4
        prefix=prefix/f'stage_{backward_stage}'
    checkpoint=prefix/f'policy_{step}.safetensors'
    with checkpoint.open('rb') as stream:
        size=struct.unpack('<Q',stream.read(8))[0]
        if size>1_000_000:raise ValueError('Unexpected checkpoint metadata size')
        metadata=json.loads(stream.read(size))['__metadata__']
    saved=json.loads(metadata['user.resolved_config_json'])
    last_grasp=max(g['start'] for g in cfg['grasps'])
    hold_start=int(cfg.get('backward_hold_source_frame',last_grasp+12))
    end_source=min(cfg['frames']-1,max(last_grasp+30,hold_start+18) if backward_stage is not None else last_grasp+22)
    start_source=[hold_start,last_grasp,max(0,last_grasp-10),0][backward_stage] if backward_stage is not None else 0
    start=round(start_source*(2*(cfg['frames']-1)-1)/(cfg['frames']-1))
    end=int(round(end_source*(2*(cfg['frames']-1)-1)/(cfg['frames']-1)))+1
    if saved['task']['motion_end_frame']!=end or saved['task'].get('motion_start_frame',0)!=start or float(metadata['user.voc_scale'])!=0.:
        raise ValueError('Require a zero-assistance-stage checkpoint of the intended opening window')
    label=f'grasp-prefix-{step}'+(f'-backward-{backward_stage}' if backward_stage is not None else '')
    output=root/'evaluation'/f'{label}-metrics.json'
    checksum=digest(checkpoint)
    if not output.exists():
        docker(['exec',cfg['native_container'],'env','V2D_WRIST_VELOCITY_GAIN=0.',
            f"V2D_REFERENCE_OVERRIDE=/episode/human_motion_data/{cfg['training_reference']}/processed/sequence_id={cfg['sequence']}/robot_name=sharpa_wave",
            f"V2D_SIM_SUBSTEPS={saved['sim']['substeps']}",'python','-u',
            '/episode/scripts/evaluate_reference_controller.py',
            'evaluation.checkpoint=/episode/'+str(checkpoint.relative_to(root)),
            'evaluation.world_count=32','logging.mode=disabled',
            'evaluation.metrics_output=/episode/'+str(output.relative_to(root)),
            f'hydra.run.dir=/episode/hydra/grasp-prefix-{step}'],root/f'logs/grasp-prefix-{step}-eval.log')
    outcome=json.loads(output.read_text())
    assert outcome['step_count']==end-start and outcome['world_count']==32
    assert outcome['motion_start_frame']==start and outcome['motion_end_frame']==end
    assert digest(checkpoint)==checksum
    report={'episode':ep,'updated_utc':utc(),'status':'diagnostic_completed',
        'run':str(root),'training_reset_settling_steps':saved['env']['reset']['voc_decay_steps'],
        'training_object_support_force_limit_N':saved['env']['voc']['max_force'],
        'training_object_support_torque_limit_Nm':saved['env']['voc']['max_torque'],
        'evaluation_object_support_force_limit_N':0.,'evaluation_object_support_torque_limit_Nm':0.,
        'full_motion_test':False,'published_as_full_rollout':False,
        'source_frames_covered':[start_source,end_source],'native_control_frames':end-start,
        'backward_stage':backward_stage,
        'checkpoint':str(checkpoint),'checkpoint_sha256':checksum,
        'native_evaluation':outcome,
        'limitation':'Opening-grasp performance alone cannot establish a full-episode improvement.'}
    save(BATCH/'improvements_20261020_grasp'/f'episode_{ep:06d}-{label}-diagnostic.json',report)
    print(json.dumps({'episode':ep,'step':step,'prefix_only':True,
        'completion_32_worlds':outcome['metrics']['chord_sr'],'source_end_frame':end_source}),flush=True)
    return report


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('episode',type=int)
    parser.add_argument('step',type=int)
    parser.add_argument('--backward-stage',type=int)
    parser.add_argument('--variant',choices=['grasp_curriculum_20261020','backward_grasp_20261020'],default='grasp_curriculum_20261020')
    args=parser.parse_args()
    evaluate(args.episode,args.step,args.backward_stage,args.variant)
