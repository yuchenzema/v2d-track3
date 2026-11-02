"""Validate promising finger corrections as complete, single-profile policies."""
from pathlib import Path
import argparse
import hashlib
import json
import traceback
from batch_review import review
from batch_native import docker
from improvement_common import BASE,BATCH,root,baseline,better,save,utc,verify_preserved
from object_motion_quality import attach
from render_improved_controller import render
from video_catalog import catalog


def validate(ep,kind='grip'):
    if kind not in ['grip','reference-wrist']:raise ValueError('Unknown controller validation kind')
    out=BATCH/'improvements_20261020_grasp'
    path=out/f'episode_{ep:06d}-{kind}-controller-result.json'
    source=json.loads((out/f'episode_{ep:06d}-grip-sweep-source-review.json').read_text())
    status=json.loads((out/f'episode_{ep:06d}-grip-sweep-status.json').read_text())
    original=baseline(ep)
    release=catalog(ep,root(ep),include_selected=True)['best']
    previous=attach(release['run'],release['selected'])
    previous_sr=release['quality']['completion_rate_against_video_estimate']
    target=BASE/'runs'/f'episode_{ep:06d}_newton_grasp_curriculum_20261020'
    cfg=json.loads((target/'episode-config.json').read_text())
    checkpoint=target/'research'/f'baseline-for-grip-{status["checkpoint_sha256"][:16]}.safetensors'
    profiles=sorted([r for r in source['profiles'] if r['passes_initial_per_object_nonregression']],
        key=lambda r:r['object_motion_quality']['score'],reverse=True)[:2]
    if kind=='reference-wrist':profiles=[{'profile':{'wrist_position':0.,'wrist_orientation':0.,'finger':1.}},
        {'profile':{'wrist_position':0.,'wrist_orientation':0.,'finger':.5}}]
    candidates=[];errors=[];selected=None;rendered=None;cohort=None;quality=None
    save(path,{'episode':ep,'status':'testing','updated_utc':utc(),'profiles':[r['profile'] for r in profiles]})
    for proposal in profiles:
        profile=proposal['profile']
        try:
            row=review(ep,original['environment_steps'],original['wrist_reference_velocity_gain'],
                original['inference_reference'],original['physics_substeps'],
                checkpoint_override='/episode/'+str(checkpoint.relative_to(target)),
                root_override=target,container_override=cfg['native_container'],
                **({'grip_bias':profile} if kind=='grip' else {'residual_profile':profile}))
            candidates.append(row)
            save(path,{'episode':ep,'status':'testing','updated_utc':utc(),'candidates':candidates,'errors':errors})
            if not better(row,previous):continue
            tag=hashlib.sha256(json.dumps(profile,sort_keys=True).encode()).hexdigest()[:10]
            destination=target/f'research/{kind}_controller_validation'/tag;destination.mkdir(parents=True,exist_ok=True)
            metrics=destination/'metrics-128.json'
            if not metrics.exists():
                variable='V2D_GRIP_BIAS' if kind=='grip' else 'V2D_RESIDUAL_PROFILE'
                helper='evaluate_grip_bias.py' if kind=='grip' else 'evaluate_residual_profile.py'
                docker(['exec',cfg['native_container'],'env',f'{variable}={json.dumps(profile,sort_keys=True)}',
                    f'V2D_WRIST_VELOCITY_GAIN={row["wrist_reference_velocity_gain"]}',
                    f'V2D_SIM_SUBSTEPS={row["physics_substeps"]}',
                    f"V2D_REFERENCE_OVERRIDE=/episode/human_motion_data/{row['inference_reference']}/processed/sequence_id={cfg['sequence']}/robot_name=sharpa_wave",
                    'python','-u','/episode/scripts/'+helper,f'evaluation.checkpoint={row["checkpoint"]}',
                    'evaluation.world_count=128','logging.mode=disabled',
                    'evaluation.metrics_output=/episode/'+str(metrics.relative_to(target)),
                    f'hydra.run.dir=/episode/hydra/{kind}-controller-{tag}-128'],destination/'evaluate-128.log')
            cohort=json.loads(metrics.read_text())
            if cohort['metrics']['chord_sr']<previous_sr-.0625:
                errors.append({'profile':profile,'rejected':'128-world completion regressed'});continue
            rendered=render(ep,row,destination,root_override=target,container_override=cfg['native_container'])
            selected=row
            q=row['object_motion_quality']
            quality={'all_internal_frame_checks_passed':q['every_object_passes_all_local_frames'] and cohort['metrics']['chord_sr']==1.,
                'completion_rate_against_video_estimate':cohort['metrics']['chord_sr'],
                'source_frames_below_internal_0_5_iou':{n:v['source_frames_below_0_5_iou'] for n,v in q['objects'].items()},
                'full_human_video_match_verified':False}
            break
        except Exception as exc:errors.append({'profile':profile,'error':str(exc),'traceback':traceback.format_exc()})
    verify_preserved(ep)
    result={'episode':ep,'status':'completed','updated_utc':utc(),'evaluation_run':str(target),
        'experiment':'Learned baseline actor with smooth declared finger flexion correction' if kind=='grip' else 'Reconstructed wrist trajectory with learned finger residuals',
        'baseline':original,'previous_best':previous,'selected':selected or previous,
        'candidates':candidates,'errors':errors,'measured_improvement':selected is not None,
        'rendered':rendered,'native_128_world_evaluation':cohort,'quality_assessment':quality,
        'full_human_motion_match_verified':False}
    save(path,result)
    print(ep,kind,'single-profile validation complete; improved',selected is not None,flush=True)
    return result


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('episode',type=int)
    parser.add_argument('--kind',choices=['grip','reference-wrist'],default='grip')
    args=parser.parse_args();validate(args.episode,args.kind)
