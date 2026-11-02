"""Search valid hand postures without modifying an active training reference."""
import argparse
import json
import shutil
from improvement_common import BASE,BATCH,save,utc,verify_preserved
from batch_native import docker


def search(ep,objects=None):
    target=BASE/'runs'/f'episode_{ep:06d}_newton_grasp_curriculum_20261020'
    cfg=json.loads((target/'episode-config.json').read_text())
    for name in ['diagnose_static_grasp.py','diagnose_grasp_pose_search.py']:
        shutil.copy2(BASE/name,target/'scripts'/name)
    output=target/'evaluation/grasp-pose-search.json'
    if not output.exists():
        extra=['env','V2D_GRASP_SEARCH_OBJECTS='+json.dumps(objects)] if objects else []
        docker(['exec',cfg['native_container'],*extra,'python','-u','/episode/scripts/diagnose_grasp_pose_search.py'],
            target/'logs/grasp-pose-search.log')
    result=json.loads(output.read_text());verify_preserved(ep)
    save(BATCH/'improvements_20261020_grasp'/f'episode_{ep:06d}-grasp-pose-diagnostic.json',{'updated_utc':utc(),**result})
    print(ep,{name:{key:v[key] for key in ['stable_hold_and_lift','static_position_drift_m','object_following_error_m']}
        for name,v in result['best_candidates'].items()},flush=True)
    return result


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('episodes',nargs='+',type=int)
    parser.add_argument('--objects',nargs='+');args=parser.parse_args()
    for episode in args.episodes:search(episode,args.objects)
