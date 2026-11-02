"""Test pickup and brushing grasps separately without changing video-derived object poses."""
import json,os,shlex,shutil,subprocess
from pathlib import Path
import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq
from scipy.spatial.transform import Rotation
from improvement_common import BASE,save,utc,verify_preserved
from batch_native import docker

def build():
    ep=30;r=BASE/'runs/episode_000030_newton_grasp_curriculum_20261020'
    assert not (r/'coherent-input-checksums.json').exists()
    cfg=json.loads((r/'episode-config.json').read_text());baseline=Path(cfg['baseline_run'])
    bc=json.loads((baseline/'episode-config.json').read_text());original=baseline/'human_motion_data'/bc['training_reference']
    reference='track3_phase_grasp_candidate';candidate=r/'human_motion_data'/reference
    if not candidate.exists():shutil.copytree(original,candidate)
    path=next(candidate.rglob('motion.parquet'));old=pq.ParquetFile(next(original.rglob('motion.parquet'))).read().to_pylist()[0]
    early=json.loads((r/'evaluation/refined-grasp-search-source-34.json').read_text())['best_candidates']['mini_sweeper']
    late=json.loads((r/'evaluation/reanchored-grasp-search.json').read_text())['best_candidates']['mini_sweeper']
    assert early['stable_hold_and_lift'] and late['stable_hold_and_lift']
    grasp=next(g for g in cfg['grasps'] if g['side']=='right')
    fits=[]
    for best,interval in [(early,[grasp['start'],56]),(late,[57,grasp['end']])]:
        assert best['finger_joint_names']==old['right_robot_finger_joint_names']
        fits.append({'side':'right','object_index':0,'object':'mini_sweeper','held_interval':interval,
            'wrist_in_object':best['wrist_in_object'],'finger_joints':best['finger_joints'],
            'template_source_frame':best['source_frame']})
    save(r/'research/phase-grasp-candidate-fit.json',{'grasps':fits,'phase_boundary_source_frame':57,
        'reason':'Source frames34..50 retain the brush in its holder; the brush clears the rack around54. Blend into the later brushing grip over54..60.'})
    image=subprocess.check_output(['sg','docker','-c',shlex.join(['docker','inspect','v2d-batch-episode32-retarget','--format','{{.Image}}'])],text=True).strip()
    shutil.copy2(BASE/'apply_collision_transitions.py',r/'scripts/apply_collision_transitions.py')
    command=['docker','run','--rm','--name','v2d-brush30-phase-fk-cpu','--ipc','host','-e','CUDA_VISIBLE_DEVICES=',
        '-e',f'V2D_TRANSITION_REFERENCE={reference}','-e','V2D_CONTACT_FIT_OUTPUT=research/phase-grasp-candidate-fit.json',
        '-e','V2D_CONTACT_APPLY_OUTPUT=research/phase-grasp-candidate-applied.json','-e','V2D_BLEND_ADJACENT_FITS=1',
        '-v',f'{r}:/episode','-v',f'{BASE}/video_to_data/robotic_grounding/source/robotic_grounding/robotic_grounding/assets:/v2d/assets:ro',
        '--entrypoint','/isaac-sim/python.sh',image,'-u','/episode/scripts/apply_collision_transitions.py']
    def fk(label):
        with (r/f'logs/{label}.log').open('w') as stream:
            subprocess.run(['sg','docker','-c',shlex.join(command)],stdout=stream,stderr=subprocess.STDOUT,check=True)
    fk('phase-grasp-fk')
    row=pq.ParquetFile(path).read().to_pylist()[0];prefix='robot_right_'
    w=np.asarray(row[prefix+'wrist_position']);q=np.asarray(row[prefix+'wrist_wxyz']);f=np.asarray(row[prefix+'finger_joints'])
    ow=np.asarray(old[prefix+'wrist_position']);oq=np.asarray(old[prefix+'wrist_wxyz']);of=np.asarray(old[prefix+'finger_joints'])
    start,end=grasp['start'],grasp['end']
    for anchor,indices,width in [(start,range(max(0,start-8),start),8),(end,range(end+1,min(cfg['frames'],end+8)),8)]:
        delta=Rotation.from_quat(q[anchor],scalar_first=True)*Rotation.from_quat(oq[anchor],scalar_first=True).inv()
        for i in indices:
            b=np.clip(1-abs(i-anchor)/width,0.,1.);b=b*b*(3.-2.*b)
            w[i]=(1-b)*ow[i]+b*(w[anchor]+delta.apply(ow[i]-ow[anchor]))
            q[i]=(Rotation.from_rotvec(delta.as_rotvec()*b)*Rotation.from_quat(oq[i],scalar_first=True)).as_quat(scalar_first=True)
            if i<anchor:
                fb=np.clip(1-abs(i-anchor)/4.,0.,1.);fb=fb*fb*(3.-2.*fb);f[i]=(1-fb)*of[i]+fb*f[anchor]
    for key,value in [('wrist_position',w),('wrist_wxyz',q),('finger_joints',f)]:row[prefix+key]=value.tolist()
    pq.write_table(pa.Table.from_pylist([row]),path,compression='zstd');fk('phase-grasp-boundaries-fk')
    row=pq.ParquetFile(path).read().to_pylist()[0]
    for key in ['object_body_position','object_body_wxyz','object_body_names','object_mesh_paths','object_urdf_paths']:
        assert np.array_equal(row[key],old[key])
    for name in ['wrist_motor_controller.py','diagnose_reconstruction_motion.py','diagnose_zero_residual.py']:
        shutil.copy2(BASE/name,r/'scripts'/name)
    label='phase-grasp-candidate-unassisted-pd'
    docker(['exec',cfg['native_container'],'env','V2D_WRIST_MOTOR_PROFILE={"kp":250,"kd":20}',
        'V2D_WRIST_SIDE_MOTOR_PROFILES='+json.dumps(cfg['wrist_side_motor_profiles']),f'V2D_DIAGNOSTIC_REFERENCE={reference}',
        f"V2D_DIAGNOSTIC_SUBSTEPS={cfg.get('physics_substeps',5)}",f'V2D_DIAGNOSTIC_LABEL={label}',
        'python','-u','/episode/scripts/diagnose_reconstruction_motion.py'],r/f'logs/{label}.log')
    env={**os.environ,'OPENBLAS_NUM_THREADS':'1','OMP_NUM_THREADS':'1','MKL_NUM_THREADS':'1'}
    for args,name in [(['assess_single_attempt.py','--states',str(r/f'evaluation/{label}-states.npz'),'--reference',reference,
                      '--output',str(r/f'evaluation/{label}-tracking.json')],'tracking'),
                      (['assess_source_fidelity.py','--states',str(r/f'evaluation/{label}-states.npz'),'--output',label+'-source-fidelity'],'source')]:
        with (r/f'logs/{label}-{name}.log').open('w') as stream:
            subprocess.run([str(r/'.venv-video/bin/python'),str(r/'scripts'/args[0]),*args[1:]],env=env,stdout=stream,stderr=subprocess.STDOUT,check=True)
    fidelity=json.loads((r/f'evaluation/{label}-source-fidelity.json').read_text());tracking=json.loads((r/f'evaluation/{label}-tracking.json').read_text())
    result={'episode':ep,'updated_utc':utc(),'diagnostic_only':True,'policy_inference':False,'approved_for_training':False,
        'reference':reference,'source_object_poses_and_camera_unchanged':True,'source_metrics':fidelity['aggregate'],
        'single_attempt_tracking':tracking,'phase_boundary_source_frame':57}
    save(r/'research/phase-grasp-candidate-motion-review.json',result);verify_preserved(ep)
    print('30 phase reference',fidelity['aggregate'],tracking['per_body'],flush=True)

if __name__=='__main__':build()
