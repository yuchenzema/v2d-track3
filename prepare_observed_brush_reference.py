"""Repair the occluded brush pickup using the original video's stationary holder interval.

Write a separate candidate. The prior reference, CAD, camera and source clock
remain available. Rebuild FK/contact fields, then independently check collision
feasibility and silhouette agreement before admitting a new training run.
"""
import json,shutil,subprocess,shlex
from pathlib import Path
import numpy as np,pyarrow as pa,pyarrow.parquet as pq
from scipy.spatial.transform import Rotation,Slerp
from improvement_common import BASE,save,utc,verify_preserved
from batch_native import _docker

R=BASE/'runs/episode_000030_newton_grasp_curriculum_20261020'
REF='track3_observed_brush_pickup_candidate'


def smooth(value):
    value=float(np.clip(value,0,1));return value*value*(3-2*value)


def build():
    target=R/'human_motion_data'/REF
    assert not target.exists()
    shutil.copytree(R/'human_motion_data/track3_phase_grasp_candidate',target)
    path=next(target.rglob('motion.parquet'));row=pq.ParquetFile(path).read().to_pylist()[0]
    P=np.array(row['object_body_position']);Q=np.array(row['object_body_wxyz']);T=len(P)
    objects=np.tile(np.eye(4),(T,2,1,1));objects[:,:,:3,3]=P
    objects[:,:,:3,:3]=Rotation.from_quat(Q.reshape(-1,4),scalar_first=True).as_matrix().reshape(T,2,3,3)
    previous=objects.copy();relative=np.linalg.inv(objects[:,1])@objects[:,0]
    anchor=relative[0].copy();end=relative[65].copy()
    for frame in range(65):
        weight=smooth((frame-50)/15)
        local=np.eye(4);local[:3,3]=(1-weight)*anchor[:3,3]+weight*end[:3,3]
        local[:3,:3]=Slerp([0,1],Rotation.from_matrix([anchor[:3,:3],end[:3,:3]]))([weight])[0].as_matrix()
        objects[frame,0]=objects[frame,1]@local
    early=json.loads((R/'evaluation/refined-grasp-search-source-34.json').read_text())['best_candidates']['mini_sweeper']
    late=json.loads((R/'evaluation/reanchored-grasp-search.json').read_text())['best_candidates']['mini_sweeper']
    w=np.array(row['robot_right_wrist_position']);q=np.array(row['robot_right_wrist_wxyz']);f=np.array(row['robot_right_finger_joints'])
    A=np.array(early['wrist_in_object']);B=np.array(late['wrist_in_object'])
    target_hand=[]
    for frame in range(26,66):
        blend=smooth((frame-54)/11);local=np.eye(4)
        local[:3,3]=(1-blend)*A[:3,3]+blend*B[:3,3]
        local[:3,:3]=Slerp([0,1],Rotation.from_matrix([A[:3,:3],B[:3,:3]]))([blend])[0].as_matrix()
        goal=objects[frame,0]@local;approach=smooth((frame-26)/8)
        w[frame]=(1-approach)*w[frame]+approach*goal[:3,3]
        q[frame]=Slerp([0,1],Rotation.from_matrix([Rotation.from_quat(q[frame],scalar_first=True).as_matrix(),goal[:3,:3]]))([approach]).as_quat(scalar_first=True)[0]
        finger=(1-blend)*np.array(early['finger_joints'])+blend*np.array(late['finger_joints'])
        close=smooth((frame-30)/4);f[frame]=(1-close)*f[frame]+close*finger
        target_hand.append(w[frame].tolist())
    row['object_body_position']=objects[:,:,:3,3].tolist()
    row['object_body_wxyz']=Rotation.from_matrix(objects[:,:,:3,:3].reshape(-1,3,3)).as_quat(scalar_first=True).reshape(T,2,4).tolist()
    row['robot_right_wrist_position']=w.tolist();row['robot_right_wrist_wxyz']=q.tolist();row['robot_right_finger_joints']=f.tolist()
    assert np.array_equal(objects[:,1],previous[:,1]) and np.array_equal(objects[65:,0],previous[65:,0])
    pq.write_table(pa.Table.from_pylist([row]),path,compression='zstd')
    save(R/'research/observed-brush-pickup-reference.json',{'updated_utc':utc(),'reference':REF,
         'diagnostic_only':True,'approved_for_training':False,'source_frames':T,
         'observed_stationary_in_holder_through_frame':50,'visible_lift_transition':[50,65],
         'source_evidence':'Original source frames20,24,28,32,34,38,42,46,50 show the brush in the rack;54..65 show extraction.',
         'reason':'Occlusion produced nonphysical pose flips during the stationary grasp. Bind the brush to the observed moving rack until extraction.',
         'changed_brush_source_frames':[0,64],'rack_poses_camera_geometry_and_source_clock_unchanged':True,
         'maximum_brush_origin_correction_m':float(np.linalg.norm(objects[:,0,:3,3]-previous[:,0,:3,3],axis=1).max())})
    review_candidate()


def review_candidate():
    for helper in ['apply_collision_transitions.py','native_reference_audit.py','run_geometry_cpu.py']:
        shutil.copy2(BASE/helper,R/'scripts'/helper)
    image=_docker(['inspect','v2d-batch-episode32-retarget','--format','{{.Image}}']).stdout.strip()
    command=['run','--rm','--name','v2d-observed-brush30-fk-cpu','--ipc','host','-e','CUDA_VISIBLE_DEVICES=',
             '-e',f'V2D_TRANSITION_REFERENCE={REF}','-e','V2D_REBUILD_ONLY=1','-e','V2D_CONTACT_APPLY_OUTPUT=research/observed-brush-fk.json',
             '-v',f'{R}:/episode','-v',f'{BASE}/video_to_data/robotic_grounding/source/robotic_grounding/robotic_grounding/assets:/v2d/assets:ro',
             '--entrypoint','/isaac-sim/python.sh',image,'-u','/episode/scripts/apply_collision_transitions.py']
    _docker(command,R/'logs/observed-brush-fk.log')
    _docker(['exec','v2d-grasp-episode30-newton','env','CUDA_VISIBLE_DEVICES=','JAX_PLATFORMS=cpu',f'V2D_AUDIT_REFERENCE={REF}',
             'python','-u','/episode/scripts/run_geometry_cpu.py','/episode/scripts/native_reference_audit.py'],R/'logs/observed-brush-collision-audit.log')
    with (R/'logs/observed-brush-source-fidelity.log').open('w') as log:
        subprocess.run([str(R/'.venv-video/bin/python'),str(BASE/'assess_physical_reference.py'),'--run',str(R),'--reference',REF,
                        '--output','observed-brush-reference-source-fidelity','--no-update-selection'],stdout=log,stderr=subprocess.STDOUT,check=True)
    verify_preserved(30)
    print('Candidate written, FK rebuilt, full-frame physical and image audits complete.',flush=True)


if __name__=='__main__':
    build()
