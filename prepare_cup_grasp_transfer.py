"""Transfer a simulated cup grasp across identical CAD assets, then validate it."""
from pathlib import Path
import argparse
import copy
import json
import shlex
import shutil
import subprocess
import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq
from scipy.spatial.transform import Rotation
from improvement_common import BASE,BATCH,save,utc,digest,verify_preserved
from fresh_improvement_run import prepare
from batch_native import docker


def transfer(ep,diagnostic_only=False):
    source=BASE/'runs/episode_000032_newton_grasp_curriculum_20261020'
    root=BASE/'runs'/f'episode_{ep:06d}_newton_grasp_curriculum_20261020'
    assert diagnostic_only or not (root/'coherent-input-checksums.json').exists(),'Use an isolated new run for an already-started learner'
    reference='track3_cup_transfer_probe' if diagnostic_only else 'track3_physical_grasp_candidate'
    fit_file='research/cup-transfer-probe-fit.json' if diagnostic_only else 'research/physical-grasp-candidate-fit.json'
    applied_file='research/cup-transfer-probe-applied.json' if diagnostic_only else 'research/physical-grasp-candidate-applied.json'
    cfg=json.loads((root/'episode-config.json').read_text())
    assert digest(root/'reconstruction/mesh_priors/blue_cup.ply')==digest(source/'reconstruction/mesh_priors/blue_cup.ply')
    body=cfg['objects'].index('blue_cup');grasp=next(g for g in cfg['grasps'] if g['object']==body and g['side']=='left')
    fit=copy.deepcopy(json.loads((source/'research/physical-grasp-candidate-fit.json').read_text())['grasps'][0])
    fit.update(object_index=body,held_interval=[grasp['start'],grasp['end']],source_template_episode=32)
    decision=root/fit_file;save(decision,{'grasps':[fit]})
    original=root/'human_motion_data'/cfg['training_reference'];candidate=root/'human_motion_data'/reference
    if not candidate.exists():shutil.copytree(original,candidate)
    shutil.copy2(BASE/'apply_collision_transitions.py',root/'scripts/apply_collision_transitions.py')
    image=subprocess.check_output(['sg','docker','-c',shlex.join(['docker','inspect','v2d-batch-episode32-retarget','--format','{{.Image}}'])],text=True).strip()
    command=['docker','run','--rm','--name',f'v2d-grasp{ep}-fk-cpu','--ipc','host','-e','CUDA_VISIBLE_DEVICES=',
        '-e',f'V2D_TRANSITION_REFERENCE={reference}',
        '-e',f'V2D_CONTACT_FIT_OUTPUT={fit_file}',
        '-e',f'V2D_CONTACT_APPLY_OUTPUT={applied_file}',
        '-v',f'{root}:/episode','-v',f'{BASE}/video_to_data/robotic_grounding/source/robotic_grounding/robotic_grounding/assets:/v2d/assets:ro',
        '--entrypoint','/isaac-sim/python.sh',image,'-u','/episode/scripts/apply_collision_transitions.py']
    def fk(label):
        with (root/f'logs/{label}.log').open('w') as stream:
            subprocess.run(['sg','docker','-c',shlex.join(command)],stdout=stream,stderr=subprocess.STDOUT,check=True)
    fk('physical-cup-transfer-fk')
    path=next(candidate.rglob('motion.parquet'));row=pq.ParquetFile(path).read().to_pylist()[0]
    old=pq.ParquetFile(next(original.rglob('motion.parquet'))).read().to_pylist()[0]
    w=np.asarray(row['robot_left_wrist_position']);q=np.asarray(row['robot_left_wrist_wxyz']);f=np.asarray(row['robot_left_finger_joints'])
    ow=np.asarray(old['robot_left_wrist_position']);oq=np.asarray(old['robot_left_wrist_wxyz']);of=np.asarray(old['robot_left_finger_joints'])
    changed=[]
    for anchor,indices,width in [(grasp['start'],range(max(0,grasp['start']-4),grasp['start']),4),
            (grasp['end'],range(grasp['end']+1,min(cfg['frames'],grasp['end']+8)),8)]:
        delta=Rotation.from_quat(q[anchor],scalar_first=True)*Rotation.from_quat(oq[anchor],scalar_first=True).inv()
        for i in indices:
            b=max(0.,1-abs(i-anchor)/width);b=b*b*(3-2*b)
            target=w[anchor]+delta.apply(ow[i]-ow[anchor]);w[i]=(1-b)*ow[i]+b*target
            q[i]=(Rotation.from_rotvec(delta.as_rotvec()*b)*Rotation.from_quat(oq[i],scalar_first=True)).as_quat(scalar_first=True)
            if i<anchor:f[i]=(1-b)*of[i]+b*f[anchor]
            changed.append(i)
    row['robot_left_wrist_position']=w.tolist();row['robot_left_wrist_wxyz']=q.tolist();row['robot_left_finger_joints']=f.tolist()
    pq.write_table(pa.Table.from_pylist([row]),path,compression='zstd');fk('physical-cup-transfer-boundaries-fk')
    row=pq.ParquetFile(path).read().to_pylist()[0]
    for key in ['object_body_position','object_body_wxyz','object_body_names','object_mesh_paths','object_urdf_paths']:
        assert np.array_equal(row[key],old[key])
    for name in ['diagnose_static_grasp.py','validate_grasp_reference.py']:
        shutil.copy2(BASE/name,root/'scripts'/name)
    docker(['exec',cfg['native_container'],'env','V2D_WRIST_MOTOR_PROFILE={"kp":250,"kd":20}',
        f'V2D_GRASP_PROBE_REFERENCE={reference}','python','-u','/episode/scripts/diagnose_static_grasp.py'],
        root/'logs/physical-cup-transfer-static-probe.log')
    sample=min(cfg['frames']-1,max(g['start'] for g in cfg['grasps'])+12)
    probe=root/f'evaluation/static-grasp-hold-lift-source-{sample}-v2-kp250-kd20-reference-{reference}.json'
    evidence=json.loads(probe.read_text())
    if diagnostic_only:
        save(root/'research/cup-transfer-probe-decision.json',{'episode':ep,'updated_utc':utc(),
            'diagnostic_only':True,'approved_for_independent_training':False,'reference':reference,
            'source_object_poses_and_camera_unchanged':True,'evidence':evidence})
        verify_preserved(ep)
        print(ep,'diagnostic cup transfer',evidence['objects'],flush=True)
        return
    if not evidence['objects']['blue_cup']['stable_hold_and_lift']:
        save(root/'research/physical-grasp-transfer-decision.json',{'episode':ep,'updated_utc':utc(),
            'approved_for_independent_training':False,'reason':'Transferred cup grasp failed the target static test','evidence':evidence})
        print(ep,'cup transfer needs further refinement',evidence['objects'],flush=True);return
    selection={'episode':ep,'updated_utc':utc(),'approved_for_independent_training':True,
        'verified_policy_improvement':False,'reference':'track3_physical_grasp_candidate',
        'wrist_motor_profile':{'kp':250.,'kd':20.},'source_object_poses_and_camera_unchanged':True,
        'source_grasp_episode':32,'identical_cup_mesh_sha256':digest(root/'reconstruction/mesh_priors/blue_cup.ply'),
        'target_static_grasp_evidence':evidence,'boundary_frames_blended':changed,
        'remaining_learning_tasks':['Continuous approach and pickup','Full source motion and release'],
        'interpretation':'Approved training experiment; no learned-policy improvement claimed before full validation'}
    save(root/'research/physical-grasp-reference-selection.json',selection)
    run,container=prepare(ep,'grasp_curriculum_20261020')
    for stage in ['prefix','full']:
        cmd=['docker','exec',container,'env','CUDA_VISIBLE_DEVICES=','JAX_PLATFORMS=cpu','V2D_VALIDATE_CONFIG_ONLY=1',
            f'V2D_GRASP_STAGE={stage}','python','-u','/episode/scripts/train_coherent_actor_transfer.py']
        with (run/f'logs/physical-cup-transfer-training-{stage}-config.log').open('w') as stream:
            subprocess.run(['sg','docker','-c',shlex.join(cmd)],stdout=stream,stderr=subprocess.STDOUT,check=True)
    verify_preserved(ep)
    priority_path=BATCH/'training-priority.json';priority=json.loads(priority_path.read_text())
    priority['held_for_reconstruction_review']=[e for e in priority.get('held_for_reconstruction_review',[]) if e!=ep]
    if run.name not in priority['allowed_run_names']:priority['allowed_run_names'].append(run.name)
    priority['updated_utc']=utc();save(priority_path,priority)
    print(ep,'cup transfer physically validated; independent training released.',flush=True)


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('episode',type=int)
    parser.add_argument('--diagnostic-only',action='store_true')
    args=parser.parse_args();transfer(args.episode,args.diagnostic_only)
