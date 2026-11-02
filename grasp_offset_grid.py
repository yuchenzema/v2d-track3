"""Run bounded physical grasp searches without editing active training inputs."""
import argparse
import json
import shutil
from improvement_common import BASE,BATCH,save,utc,verify_preserved
from batch_native import docker


def search(ep):
    target=BASE/'runs'/f'episode_{ep:06d}_newton_grasp_curriculum_20261020'
    cfg=json.loads((target/'episode-config.json').read_text())
    for name in ['diagnose_static_grasp.py','diagnose_grasp_offset_grid.py']:
        shutil.copy2(BASE/name,target/'scripts'/name)
    output=target/'evaluation/grasp-offset-grid.json'
    if not output.exists():
        docker(['exec',cfg['native_container'],'python','-u','/episode/scripts/diagnose_grasp_offset_grid.py'],
            target/'logs/grasp-offset-grid.log')
    result=json.loads(output.read_text());verify_preserved(ep)
    save(BATCH/'improvements_20261020_grasp'/f'episode_{ep:06d}-grasp-offset-diagnostic.json',{'updated_utc':utc(),**result})
    for name in cfg['objects']:
        rows=sorted(result['profiles'],key=lambda row:row['objects'][name]['diagnostic_cost'])
        print(ep,name,'stable',sum(row['objects'][name]['stable_hold_and_lift'] for row in rows),'best',rows[0]['profile'],rows[0]['objects'][name],flush=True)
    return result


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('episodes',nargs='+',type=int)
    for episode in parser.parse_args().episodes:search(episode)
