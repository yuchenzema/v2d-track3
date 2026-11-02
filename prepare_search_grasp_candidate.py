"""Build and assess a reconstructed grasp candidate; keep it out of training until reviewed."""
from pathlib import Path
import argparse
import json
import os
import shlex
import shutil
import subprocess
import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq
from scipy.spatial.transform import Rotation
from improvement_common import BASE,BATCH,save,utc,digest,verify_preserved
from batch_native import docker


def prepare(ep,refined=False,diagnostic_only=False,transfer_brush=False,reanchored=False,pickup=False):
    root=BASE/'runs'/f'episode_{ep:06d}_newton_grasp_curriculum_20261020'
    assert diagnostic_only or not (root/'coherent-input-checksums.json').exists(),'Preserve a started learner; use another experiment'
    cfg=json.loads((root/'episode-config.json').read_text())
    evidence=json.loads((root/('evaluation/refined-grasp-search.json' if refined else 'evaluation/grasp-pose-search.json')).read_text())
    data=None if refined else np.load(root/'evaluation/grasp-pose-search.npz')
    original=root/'human_motion_data'/cfg['training_reference'];reference='track3_refined_grasp_candidate' if refined else 'track3_physical_grasp_candidate'
    fit_file='research/refined-grasp-candidate-fit.json' if refined else 'research/physical-grasp-candidate-fit.json'
    if transfer_brush:
        assert ep==30 and refined
        source=BASE/'runs/episode_000019_newton_grasp_curriculum_20261020'
        assert digest(root/'reconstruction/mesh_priors/mini_sweeper.ply')==digest(source/'reconstruction/mesh_priors/mini_sweeper.ply')
        evidence=json.loads((source/'evaluation/refined-grasp-search.json').read_text())
        evidence['best_candidates']={'mini_sweeper':evidence['best_candidates']['mini_sweeper']}
        reference='track3_transferred_grasp_candidate';fit_file='research/transferred-grasp-candidate-fit.json'
    if reanchored:
        assert refined and not transfer_brush
        evidence=json.loads((root/'evaluation/reanchored-grasp-search.json').read_text())
        reference='track3_reanchored_grasp_candidate';fit_file='research/reanchored-grasp-candidate-fit.json'
    if pickup:
        assert ep==30 and refined
        evidence=json.loads((root/'evaluation/refined-grasp-search-source-34.json').read_text())
        reference='track3_pickup_grasp_candidate';fit_file='research/pickup-grasp-candidate-fit.json'
        baseline_root=Path(cfg['baseline_run']);baseline_cfg=json.loads((baseline_root/'episode-config.json').read_text())
        original=baseline_root/'human_motion_data'/baseline_cfg['training_reference']
    candidate=root/'human_motion_data'/reference
    if not candidate.exists():shutil.copytree(original,candidate)
    old=pq.ParquetFile(next(original.rglob('motion.parquet'))).read().to_pylist()[0]
    if ep==20 and refined:
        cup=json.loads((root/'research/cup-transfer-probe-fit.json').read_text())['grasps'][0]
        evidence['best_candidates']['blue_cup']={**cup,'finger_joint_names':old['left_robot_finger_joint_names'],
            'source_frame':cup.get('template_source_frame',0)}
    fits=[]
    for name,best in evidence['best_candidates'].items():
        if ep==30 and name!='mini_sweeper':continue # Preserve the rack's existing grasp and stationary motion.
        side=best['side'];body=cfg['objects'].index(name)
        grasp=next(g for g in cfg['grasps'] if g['side']==side and g['object']==body)
        assert best['finger_joint_names']==old[f'{side}_robot_finger_joint_names']
        if refined:
            local=np.asarray(best['wrist_in_object'])
        else:
            pose=data[name+'_initial_body_poses'][best['local_object_body_id']]
            obj=np.eye(4);obj[:3,:3]=Rotation.from_quat(pose[3:]).as_matrix();obj[:3,3]=pose[:3]
            wrist=np.eye(4);wrist[:3,:3]=best['reference_wrist_rotation']
            wrist[:3,3]=np.asarray(best['reference_wrist_position'])+np.asarray(best['world_inward_direction'])*best['inward_m']
            local=np.linalg.inv(obj)@wrist
        fits.append({'side':side,'object_index':body,'object':name,'held_interval':[grasp['start'],grasp['end']],
            'wrist_in_object':local.tolist(),'finger_joints':best['finger_joints'],
            'template_source_frame':best['source_frame'] if refined else evidence['source_frame'],
            'source':'Physical wrist/finger search; source objects/camera unchanged'})
    save(root/fit_file,{'grasps':fits})
    for helper in ['apply_collision_transitions.py','wrist_motor_controller.py','diagnose_reconstruction_motion.py','diagnose_zero_residual.py']:
        shutil.copy2(BASE/helper,root/'scripts'/helper)
    image=subprocess.check_output(['sg','docker','-c',shlex.join(['docker','inspect','v2d-batch-episode32-retarget','--format','{{.Image}}'])],text=True).strip()
    command=['docker','run','--rm','--name',f'v2d-grasp{ep}-fk-cpu','--ipc','host','-e','CUDA_VISIBLE_DEVICES=',
        '-e',f'V2D_TRANSITION_REFERENCE={reference}','-e',f'V2D_CONTACT_FIT_OUTPUT={fit_file}',
        '-e','V2D_CONTACT_APPLY_OUTPUT=research/physical-grasp-candidate-applied.json',
        '-v',f'{root}:/episode','-v',f'{BASE}/video_to_data/robotic_grounding/source/robotic_grounding/robotic_grounding/assets:/v2d/assets:ro',
        '--entrypoint','/isaac-sim/python.sh',image,'-u','/episode/scripts/apply_collision_transitions.py']
    def fk(label):
        with (root/f'logs/{label}.log').open('w') as stream:
            subprocess.run(['sg','docker','-c',shlex.join(command)],stdout=stream,stderr=subprocess.STDOUT,check=True)
    fk('physical-search-grasp-fk')
    path=next(candidate.rglob('motion.parquet'));row=pq.ParquetFile(path).read().to_pylist()[0]
    for fit in fits:
        side=fit['side'];start,end=fit['held_interval'];prefix='robot_'+side+'_'
        w=np.asarray(row[prefix+'wrist_position']);q=np.asarray(row[prefix+'wrist_wxyz']);f=np.asarray(row[prefix+'finger_joints'])
        ow=np.asarray(old[prefix+'wrist_position']);oq=np.asarray(old[prefix+'wrist_wxyz']);of=np.asarray(old[prefix+'finger_joints'])
        for anchor,indices,width in [(start,range(max(0,start-4),start),4),(end,range(end+1,min(cfg['frames'],end+8)),8)]:
            delta=Rotation.from_quat(q[anchor],scalar_first=True)*Rotation.from_quat(oq[anchor],scalar_first=True).inv()
            for i in indices:
                b=max(0.,1-abs(i-anchor)/width);b=b*b*(3-2*b)
                w[i]=(1-b)*ow[i]+b*(w[anchor]+delta.apply(ow[i]-ow[anchor]))
                q[i]=(Rotation.from_rotvec(delta.as_rotvec()*b)*Rotation.from_quat(oq[i],scalar_first=True)).as_quat(scalar_first=True)
                if i<anchor:f[i]=(1-b)*of[i]+b*f[anchor]
        row[prefix+'wrist_position']=w.tolist();row[prefix+'wrist_wxyz']=q.tolist();row[prefix+'finger_joints']=f.tolist()
    pq.write_table(pa.Table.from_pylist([row]),path,compression='zstd');fk('physical-search-grasp-boundaries-fk')
    row=pq.ParquetFile(path).read().to_pylist()[0]
    for key in ['object_body_position','object_body_wxyz','object_body_names','object_mesh_paths','object_urdf_paths']:
        assert np.array_equal(row[key],old[key])
    label='pickup-grasp-candidate-unassisted-pd' if pickup else 'reanchored-grasp-candidate-unassisted-pd' if reanchored else 'transferred-grasp-candidate-unassisted-pd' if transfer_brush else 'refined-grasp-candidate-unassisted-pd' if refined else 'physical-grasp-candidate-unassisted-pd'
    docker(['exec',cfg['native_container'],'env','V2D_WRIST_MOTOR_PROFILE={"kp":250,"kd":20}',
        'V2D_WRIST_SIDE_MOTOR_PROFILES='+json.dumps(cfg.get('wrist_side_motor_profiles',{})),
        f'V2D_DIAGNOSTIC_REFERENCE={reference}',f'V2D_DIAGNOSTIC_SUBSTEPS={cfg.get("physics_substeps",5)}',
        f'V2D_DIAGNOSTIC_LABEL={label}','python','-u','/episode/scripts/diagnose_reconstruction_motion.py'],root/f'logs/{label}.log')
    py=BASE/'runs/episode_000005_newton_20261015/.venv-video/bin/python'
    environment={**os.environ,'OPENBLAS_NUM_THREADS':'1','OMP_NUM_THREADS':'1','MKL_NUM_THREADS':'1'}
    for args,name in [(['assess_single_attempt.py','--states',str(root/f'evaluation/{label}-states.npz'),'--reference',reference,
                        '--output',str(root/f'evaluation/{label}-tracking.json')],'tracking'),
                      (['assess_source_fidelity.py','--states',str(root/f'evaluation/{label}-states.npz'),'--output',label+'-source-fidelity'],'source')]:
        with (root/f'logs/{label}-{name}.log').open('w') as stream:
            subprocess.run([str(py),str(root/'scripts'/args[0]),*args[1:]],env=environment,stdout=stream,stderr=subprocess.STDOUT,check=True)
    verify_preserved(ep)
    frozen=root/'coherent-input-checksums.json'
    if frozen.exists():
        import hashlib
        for name,checksum in json.loads(frozen.read_text()).items():
            assert hashlib.sha256((root/name).read_bytes()).hexdigest()==checksum,('Frozen input changed',name)
    fidelity=json.loads((root/f'evaluation/{label}-source-fidelity.json').read_text())
    tracking=json.loads((root/f'evaluation/{label}-tracking.json').read_text())
    result={'episode':ep,'updated_utc':utc(),'diagnostic_only':True,'approved_for_training':False,
        'reference':reference,'wrist_motor_profile':{'kp':250.,'kd':20.},'source_object_poses_and_camera_unchanged':True,
        'wrist_side_motor_profiles':cfg.get('wrist_side_motor_profiles',{}),
        'source_metrics':fidelity['aggregate'],'single_attempt_tracking':tracking,'hand_grasps_updated':[f['side'] for f in fits]}
    review_file='research/pickup-grasp-candidate-motion-review.json' if pickup else 'research/reanchored-grasp-candidate-motion-review.json' if reanchored else 'research/transferred-grasp-candidate-motion-review.json' if transfer_brush else 'research/refined-grasp-candidate-motion-review.json' if refined else 'research/physical-grasp-candidate-motion-review.json'
    save(root/review_file,result)
    print(ep,'candidate full-motion review',fidelity['aggregate']['objects'],tracking['per_body'],flush=True)
    return result


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('episodes',nargs='+',type=int)
    parser.add_argument('--refined',action='store_true');parser.add_argument('--diagnostic-only',action='store_true')
    parser.add_argument('--transfer-brush',action='store_true');parser.add_argument('--reanchored',action='store_true')
    parser.add_argument('--pickup',action='store_true');args=parser.parse_args()
    for episode in args.episodes:prepare(episode,args.refined,args.diagnostic_only,args.transfer_brush,args.reanchored,args.pickup)
