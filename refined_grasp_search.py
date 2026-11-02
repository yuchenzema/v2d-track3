"""Run independent physical wrist/finger refinement diagnostics."""
import argparse,json,shutil
from improvement_common import BASE,verify_preserved
from batch_native import docker

def search(ep,source_frame=None,objects=None,variant='grasp_curriculum_20261020',reference_seed=False,protect_other_objects=False):
    assert variant in ['grasp_curriculum_20261020','backward_grasp_20261020']
    root=BASE/'runs'/f'episode_{ep:06d}_newton_{variant}'
    cfg=json.loads((root/'episode-config.json').read_text())
    for name in ['diagnose_static_grasp.py','diagnose_refined_grasp_search.py','grasp_search_protocol.py','rigid_pose_collision_audit.py']:
        shutil.copy2(BASE/name,root/'scripts'/name)
    label='refined-grasp-search'+(f'-source-{source_frame}' if source_frame is not None else '')
    if not (root/f'evaluation/{label}.json').exists():
        extra=[f'V2D_STATIC_GRASP_SOURCE_FRAME={source_frame}'] if source_frame is not None else []
        if objects:extra+=['V2D_GRASP_SEARCH_OBJECTS='+json.dumps(objects)]
        if reference_seed:extra+=['V2D_GRASP_USE_REFERENCE_SEED=1']
        if protect_other_objects:extra+=['V2D_PROTECT_OTHER_OBJECTS=1']
        docker(['exec',cfg['native_container'],'env',f'V2D_REFINED_GRASP_LABEL={label}',*extra,'python','-u','/episode/scripts/diagnose_refined_grasp_search.py'],root/f'logs/{label}.log')
    verify_preserved(ep)
    assert 'nefc overflow' not in (root/f'logs/{label}.log').read_text()
    result=json.loads((root/f'evaluation/{label}.json').read_text())
    print(ep,{n:{k:v[k] for k in ['stable_hold_and_lift','static_position_drift_m','object_following_error_m','object_orientation_change_rad']}
        for n,v in result['best_candidates'].items()},flush=True)

if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('episodes',type=int,nargs='+')
    parser.add_argument('--source-frame',type=int);parser.add_argument('--objects',nargs='+')
    parser.add_argument('--variant',choices=['grasp_curriculum_20261020','backward_grasp_20261020'],default='grasp_curriculum_20261020')
    parser.add_argument('--reference-seed',action='store_true');parser.add_argument('--protect-other-objects',action='store_true');args=parser.parse_args()
    for episode in args.episodes:search(episode,args.source_frame,args.objects,args.variant,args.reference_seed,args.protect_other_objects)
