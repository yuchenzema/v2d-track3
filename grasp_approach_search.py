"""Prepare physical grasp fits and test full source-timed hand approaches."""
import argparse,copy,json,os,shutil
from improvement_common import BASE,digest,save,utc,verify_preserved
from batch_native import docker

def search(ep,angular_kp=25.):
    root=BASE/'runs'/f'episode_{ep:06d}_newton_grasp_curriculum_20261020'
    cfg=json.loads((root/'episode-config.json').read_text())
    if ep==30:
        source=BASE/'runs/episode_000019_newton_grasp_curriculum_20261020'
        assert digest(root/'reconstruction/mesh_priors/mini_sweeper.ply')==digest(source/'reconstruction/mesh_priors/mini_sweeper.ply')
        data=json.loads((source/'evaluation/refined-grasp-search.json').read_text())
        candidates={'mini_sweeper':copy.deepcopy(data['best_candidates']['mini_sweeper'])}
    else:candidates=json.loads((root/'evaluation/refined-grasp-search.json').read_text())['best_candidates']
    fits=[]
    for name,best in candidates.items():
        body=cfg['objects'].index(name);side=best['side'];grasp=next(g for g in cfg['grasps'] if g['side']==side and g['object']==body)
        fits.append({'side':side,'object_index':body,'object':name,'held_interval':[grasp['start'],grasp['end']],
                     'wrist_in_object':best['wrist_in_object'],'finger_joints':best['finger_joints'],
                     'stable_static_grasp':best['stable_hold_and_lift'],'source_template_episode':19 if ep==30 else ep})
    if ep==20:fits.extend(json.loads((root/'research/cup-transfer-probe-fit.json').read_text())['grasps'])
    save(root/'research/planned-grasp-fits.json',{'grasps':fits})
    for name in ['diagnose_static_grasp.py','diagnose_grasp_approach_search.py']:shutil.copy2(BASE/name,root/'scripts'/name)
    label='grasp-approach-search'+(f'-angular-{angular_kp:g}' if angular_kp!=25. else '')
    if not (root/f'evaluation/{label}.json').exists():
        save(root/'research/grasp-refinement-progress.json',{'episode':ep,'updated_utc':utc(),'pid':os.getpid(),
             'status':'checking_approach','description':'Checking grasp and pickup motion','experiment':label,
             'verified_policy_improvement':False})
        docker(['exec',cfg['native_container'],'env',f'V2D_DIAGNOSTIC_WRIST_ANGULAR_KP={angular_kp}',
                f'V2D_APPROACH_SEARCH_LABEL={label}','python','-u','/episode/scripts/diagnose_grasp_approach_search.py'],root/f'logs/{label}.log')
    verify_preserved(ep)
    log=(root/f'logs/{label}.log').read_text();assert 'nefc overflow' not in log
    data=json.loads((root/f'evaluation/{label}.json').read_text())
    save(root/'research/grasp-refinement-progress.json',{'episode':ep,'updated_utc':utc(),'pid':os.getpid(),
         'status':'awaiting_review','description':'Reviewing grasp and pickup motion','experiment':label,
         'verified_policy_improvement':False})
    print(ep,angular_kp,'best approach',min(data['trials'],key=lambda x:x['cost']),flush=True)

if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('episodes',type=int,nargs='+')
    parser.add_argument('--angular-kp',type=float,default=25.);args=parser.parse_args()
    for episode in args.episodes:search(episode,args.angular_kp)
