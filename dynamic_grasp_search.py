"""Run diagnostic grasp searches over complete source-timed physical attempts."""
import argparse
import json
import shutil
from improvement_common import BASE,verify_preserved
from batch_native import docker


def search(ep):
    root=BASE/'runs'/f'episode_{ep:06d}_newton_grasp_curriculum_20261020'
    cfg=json.loads((root/'episode-config.json').read_text())
    assert not (root/'coherent-input-checksums.json').exists(), 'Use a separate experiment for trained references'
    for name in ['diagnose_static_grasp.py','diagnose_dynamic_grasp_search.py']:
        shutil.copy2(BASE/name,root/'scripts'/name)
    if not (root/'evaluation/dynamic-grasp-search.json').exists():
        docker(['exec',cfg['native_container'],'python','-u','/episode/scripts/diagnose_dynamic_grasp_search.py'],
               root/'logs/dynamic-grasp-search.log')
    verify_preserved(ep)
    report=json.loads((root/'evaluation/dynamic-grasp-search.json').read_text())
    print(ep,{name:{'baseline':x['trials'][0], 'best':min(x['trials'],key=lambda y:y['target_cost'])}
              for name,x in report['searches'].items()},flush=True)


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('episodes',type=int,nargs='+')
    for episode in parser.parse_args().episodes:search(episode)
