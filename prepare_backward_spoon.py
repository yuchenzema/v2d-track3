"""Build an independent cup/spoon reference with a physically tested spoon grasp."""
import argparse,json,os,shlex,shutil,subprocess
import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq
from scipy.spatial.transform import Rotation
from improvement_common import BASE,BATCH,save,utc,verify_preserved
from fresh_improvement_run import prepare
from batch_native import docker

def build(ep):
    original_run=BASE/'runs'/f'episode_{ep:06d}_newton_grasp_curriculum_20261020'
    previous=json.loads((original_run/'episode-config.json').read_text())
    evidence=json.loads((original_run/'evaluation/refined-grasp-search.json').read_text())
    best=evidence['best_candidates']['wooden_spoon'];assert best['stable_hold_and_lift']
    r,container=prepare(ep,'backward_grasp_20261020')
    assert not (r/'coherent-input-checksums.json').exists()
    cfg=json.loads((r/'episode-config.json').read_text())
    reference='track3_refined_spoon_grasp_candidate';candidate=r/'human_motion_data'/reference
    if not candidate.exists():shutil.copytree(original_run/'human_motion_data'/previous['training_reference'],candidate)
    path=next(candidate.rglob('motion.parquet'));old=pq.ParquetFile(path).read().to_pylist()[0]
    side=best['side'];body=cfg['objects'].index('wooden_spoon')
    grasp=next(g for g in cfg['grasps'] if g['side']==side and g['object']==body)
    assert best['finger_joint_names']==old[side+'_robot_finger_joint_names']
    fit={'side':side,'object_index':body,'object':'wooden_spoon','held_interval':[grasp['start'],grasp['end']],
         'wrist_in_object':best['wrist_in_object'],'finger_joints':best['finger_joints'],'template_source_frame':best['source_frame']}
    save(r/'research/refined-spoon-grasp-fit.json',{'grasps':[fit]})
    for name in ['apply_collision_transitions.py','diagnose_static_grasp.py']:shutil.copy2(BASE/name,r/'scripts'/name)
    image=subprocess.check_output(['sg','docker','-c',shlex.join(['docker','inspect','v2d-batch-episode32-retarget','--format','{{.Image}}'])],text=True).strip()
    command=['docker','run','--rm','--name',f'v2d-backward-spoon{ep}-fk-cpu','--ipc','host','-e','CUDA_VISIBLE_DEVICES=',
        '-e',f'V2D_TRANSITION_REFERENCE={reference}','-e','V2D_CONTACT_FIT_OUTPUT=research/refined-spoon-grasp-fit.json',
        '-e','V2D_CONTACT_APPLY_OUTPUT=research/refined-spoon-grasp-applied.json',
        '-v',f'{r}:/episode','-v',f'{BASE}/video_to_data/robotic_grounding/source/robotic_grounding/robotic_grounding/assets:/v2d/assets:ro',
        '--entrypoint','/isaac-sim/python.sh',image,'-u','/episode/scripts/apply_collision_transitions.py']
    def fk(label):
        with (r/f'logs/{label}.log').open('w') as stream:
            subprocess.run(['sg','docker','-c',shlex.join(command)],stdout=stream,stderr=subprocess.STDOUT,check=True)
    fk('refined-spoon-grasp-fk')
    row=pq.ParquetFile(path).read().to_pylist()[0];prefix='robot_'+side+'_'
    w=np.asarray(row[prefix+'wrist_position']);q=np.asarray(row[prefix+'wrist_wxyz']);f=np.asarray(row[prefix+'finger_joints'])
    ow=np.asarray(old[prefix+'wrist_position']);oq=np.asarray(old[prefix+'wrist_wxyz']);of=np.asarray(old[prefix+'finger_joints'])
    start,end=grasp['start'],grasp['end']
    for anchor,indices,width in [(start,range(max(0,start-8),start),8),(end,range(end+1,min(cfg['frames'],end+8)),8)]:
        delta=Rotation.from_quat(q[anchor],scalar_first=True)*Rotation.from_quat(oq[anchor],scalar_first=True).inv()
        for i in indices:
            b=np.clip(1-abs(i-anchor)/width,0.,1.);b=b*b*(3.-2.*b)
            w[i]=(1-b)*ow[i]+b*(w[anchor]+delta.apply(ow[i]-ow[anchor]))
            q[i]=(Rotation.from_rotvec(delta.as_rotvec()*b)*Rotation.from_quat(oq[i],scalar_first=True)).as_quat(scalar_first=True)
            fb=np.clip(1-abs(i-anchor)/(4 if i<anchor else 8),0.,1.);fb=fb*fb*(3.-2.*fb)
            f[i]=(1-fb)*of[i]+fb*f[anchor]
    for key,value in [('wrist_position',w),('wrist_wxyz',q),('finger_joints',f)]:row[prefix+key]=value.tolist()
    pq.write_table(pa.Table.from_pylist([row]),path,compression='zstd');fk('refined-spoon-grasp-boundaries-fk')
    row=pq.ParquetFile(path).read().to_pylist()[0]
    for key in ['object_body_position','object_body_wxyz','object_body_names','object_mesh_paths','object_urdf_paths']:
        assert np.array_equal(row[key],old[key])
    docker(['exec',container,'env','V2D_WRIST_MOTOR_PROFILE={"kp":250,"kd":20}',
        f'V2D_GRASP_PROBE_REFERENCE={reference}',f"V2D_STATIC_GRASP_SOURCE_FRAME={best['source_frame']}",
        'V2D_STATIC_GRASP_LIFT_SIDES=["right"]','python','-u','/episode/scripts/diagnose_static_grasp.py'],r/'logs/refined-spoon-reference-static-probe.log')
    probe_path=r/f"evaluation/static-grasp-hold-lift-source-{best['source_frame']}-v2-kp250-kd20-reference-{reference}-lift-right.json"
    probe=json.loads(probe_path.read_text())
    assert probe['objects']['wooden_spoon']['stable_hold_and_lift'], 'Rebuilt spoon grasp needs further refinement'
    selection={'episode':ep,'updated_utc':utc(),'approved_for_independent_training':True,'verified_policy_improvement':False,
        'reference':reference,'wrist_motor_profile':{'kp':250.,'kd':20.},'training_curriculum':'backward',
        'training_reset_protocol':'explicit-equivalent-zero-support','reset_settling_steps':0,
        'training_object_force_limit_N':0.,'training_object_torque_limit_Nm':0.,
        'source_object_poses_and_camera_unchanged':True,'target_static_grasp_evidence':evidence,
        'rebuilt_reference_probe':probe,'preserved_cup_grasp_from':str(original_run),
        'remaining_learning_tasks':['Continuous spoon acquisition','Cup pickup','Full stirring trajectory and placement'],
        'interpretation':'Holding feasibility only; require full learned-policy validation before publishing an improvement'}
    save(r/'research/physical-grasp-reference-selection.json',selection)
    r,container=prepare(ep,'backward_grasp_20261020')
    for stage in ['prefix','full']:
        args=['docker','exec',container,'env','CUDA_VISIBLE_DEVICES=','JAX_PLATFORMS=cpu','V2D_VALIDATE_CONFIG_ONLY=1',
              f'V2D_GRASP_STAGE={stage}','python','-u','/episode/scripts/train_coherent_actor_transfer.py']
        with (r/f'logs/backward-spoon-{stage}-configuration.log').open('w') as stream:
            subprocess.run(['sg','docker','-c',shlex.join(args)],stdout=stream,stderr=subprocess.STDOUT,check=True)
    verify_preserved(ep)
    priority_path=BATCH/'training-priority.json';priority=json.loads(priority_path.read_text())
    for key in ['additional_allowed_run_names','allowed_run_names']:
        priority.setdefault(key,[])
        if r.name not in priority[key]:priority[key].append(r.name)
    priority['updated_utc']=utc();save(priority_path,priority)
    registry=BATCH/'improvements_20261020_backward/episodes.json'
    rows=json.loads(registry.read_text()) if registry.exists() else {'episodes':[20]}
    if ep not in rows['episodes']:rows['episodes'].append(ep)
    rows['updated_utc']=utc();save(registry,rows)
    print(ep,'new spoon reference validated and queued',probe['objects'],flush=True)

if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('episode',type=int)
    build(parser.parse_args().episode)
