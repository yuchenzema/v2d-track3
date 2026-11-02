"""Resolve the symmetric brush yaw branch to avoid an artificial180-degree wrist turn."""
import json,shutil
import numpy as np,pyarrow as pa,pyarrow.parquet as pq
from scipy.spatial.transform import Rotation,Slerp
from improvement_common import BASE,save,utc
from batch_native import _docker,docker

R=BASE/'runs/episode_000030_newton_stable_brush_20261021';REF='track3_consistent_brush_pickup_candidate'


def main():
    target=R/'human_motion_data'/REF;assert not target.exists()
    shutil.copytree(R/'human_motion_data/track3_clearance_brush_pickup_candidate',target)
    path=next(target.rglob('motion.parquet'));row=pq.ParquetFile(path).read().to_pylist()[0]
    P=np.array(row['object_body_position']);Q=np.array(row['object_body_wxyz']);T=len(P);oldQ=Q.copy()
    before=float((Rotation.from_quat(Q[50,0],scalar_first=True).inv()*Rotation.from_quat(Q[65,0],scalar_first=True)).magnitude())
    flip=Rotation.from_rotvec([0,0,np.pi])
    for i in range(56):Q[i,0]=(Rotation.from_quat(Q[i,0],scalar_first=True)*flip).as_quat(scalar_first=True)
    for i in range(56,65):
        weight=(i-55)/10;weight=weight*weight*(3-2*weight)
        Q[i,0]=Slerp([0,1],Rotation.from_quat([Q[55,0],oldQ[65,0]],scalar_first=True))([weight]).as_quat(scalar_first=True)[0]
    O=np.tile(np.eye(4),(T,1,1));O[:,:3,3]=P[:,0];O[:,:3,:3]=Rotation.from_quat(Q[:,0],scalar_first=True).as_matrix()
    w=np.array(row['robot_right_wrist_position']);q=np.array(row['robot_right_wrist_wxyz']);f=np.array(row['robot_right_finger_joints'])
    H=np.eye(4);H[:3,3]=w[34];H[:3,:3]=Rotation.from_quat(q[34],scalar_first=True).as_matrix();local=np.linalg.inv(O[34])@H
    oldw=w.copy();oldq=q.copy()
    for i in range(34,254):
        H=O[i]@local;weight=float(np.clip((253-i)/8,0,1));weight=weight*weight*(3-2*weight)
        w[i]=(1-weight)*oldw[i]+weight*H[:3,3]
        q[i]=Slerp([0,1],Rotation.from_matrix([Rotation.from_quat(oldq[i],scalar_first=True).as_matrix(),H[:3,:3]]))([weight]).as_quat(scalar_first=True)[0]
    row['object_body_wxyz']=Q.tolist();row['robot_right_wrist_position']=w.tolist();row['robot_right_wrist_wxyz']=q.tolist();pq.write_table(pa.Table.from_pylist([row]),path,compression='zstd')
    after=float((Rotation.from_quat(Q[50,0],scalar_first=True).inv()*Rotation.from_quat(Q[65,0],scalar_first=True)).magnitude())
    assert after<before
    image=_docker(['inspect','v2d-batch-episode32-retarget','--format','{{.Image}}']).stdout.strip()
    _docker(['run','--rm','--name','v2d-consistent-brush30-fk-cpu','--ipc','host','-e','CUDA_VISIBLE_DEVICES=',
        '-e',f'V2D_TRANSITION_REFERENCE={REF}','-e','V2D_REBUILD_ONLY=1','-e','V2D_CONTACT_APPLY_OUTPUT=research/consistent-brush-fk.json',
        '-v',f'{R}:/episode','-v',f'{BASE}/video_to_data/robotic_grounding/source/robotic_grounding/robotic_grounding/assets:/v2d/assets:ro',
        '--entrypoint','/isaac-sim/python.sh',image,'-u','/episode/scripts/apply_collision_transitions.py'],R/'logs/consistent-brush-fk.log')
    cfg=json.loads((R/'episode-config.json').read_text());container=cfg['native_container'];label='consistent-brush-untrained-feasibility'
    docker(['exec',container,'env',f'V2D_DIAGNOSTIC_REFERENCE={REF}','V2D_DIAGNOSTIC_SUBSTEPS=20',f'V2D_DIAGNOSTIC_LABEL={label}',
        'V2D_WRIST_MOTOR_PROFILE={"kp":250,"kd":20}','V2D_WRIST_SIDE_MOTOR_PROFILES={"left":{"kp":25,"kd":6}}',
        'python','-u','/episode/scripts/diagnose_reconstruction_motion.py'],R/f'logs/{label}.log')
    save(R/'research/consistent-brush-grip-reference.json',{'updated_utc':utc(),'reference':REF,'diagnostic_only':True,
         'extraction_object_and_wrist_rotation_before_deg':float(np.degrees(before)),
         'extraction_object_and_wrist_rotation_after_deg':float(np.degrees(after)),
         'source_hand_guided_extraction_rotation_deg':48.1,'branch_choice_reason':'Symmetric initial object orientation disagreed with the later hand-guided branch and imposed an unobserved half-turn.',
         'positions_camera_CAD_and_source_clock_unchanged':True,'object_assistance':0.,'policy_improvement_verified':False})
    print('Consistent brush branch physical feasibility tested:',np.degrees(before),'→',np.degrees(after),'degrees',flush=True)


if __name__=='__main__':main()
