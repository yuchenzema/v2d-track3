"""Run isolated, explicitly non-policy grasp stability diagnostics."""
import argparse
import json
import shutil
from improvement_common import BASE,BATCH,save,utc,verify_preserved
from batch_native import docker


def probe(ep,motor_profile=None):
    target=BASE/'runs'/f'episode_{ep:06d}_newton_grasp_curriculum_20261020'
    cfg=json.loads((target/'episode-config.json').read_text())
    source=min(cfg['frames']-1,max(g['start'] for g in cfg['grasps'])+12)
    suffix=f"-kp{motor_profile['kp']:g}-kd{motor_profile['kd']:g}" if motor_profile else ''
    output=target/f'evaluation/static-grasp-hold-lift-source-{source}-v2{suffix}.json'
    shutil.copy2(BASE/'diagnose_static_grasp.py',target/'scripts/diagnose_static_grasp.py')
    if not output.exists():
        docker(['exec',cfg['native_container'],'env',*([f'V2D_WRIST_MOTOR_PROFILE={json.dumps(motor_profile)}'] if motor_profile else []),
            'python','-u','/episode/scripts/diagnose_static_grasp.py'],
            target/f'logs/static-grasp-hold-lift-v2{suffix}.log')
    result=json.loads(output.read_text());verify_preserved(ep)
    save(BATCH/'improvements_20261020_grasp'/f'episode_{ep:06d}-static-grasp-v2{suffix}-diagnostic.json',{'updated_utc':utc(),**result})
    print(ep,{name:{k:v[k] for k in ['stable_hold_and_lift','static_position_drift_m','object_following_error_m']}
        for name,v in result['objects'].items()},flush=True)
    return result


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('episodes',nargs='+',type=int)
    parser.add_argument('--kp',type=float);parser.add_argument('--kd',type=float)
    args=parser.parse_args()
    if (args.kp is None)!=(args.kd is None):parser.error('Provide both --kp and --kd')
    profile={'kp':args.kp,'kd':args.kd} if args.kp is not None else None
    for episode in args.episodes:probe(episode,profile)
