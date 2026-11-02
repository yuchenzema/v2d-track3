"""Retarget the continuous video grasp to the physically tested wooden handle grip."""
import json,shutil,subprocess
import numpy as np,pyarrow as pa,pyarrow.parquet as pq
from scipy.spatial.transform import Rotation,Slerp
from improvement_common import BASE,save,utc
from batch_native import _docker,docker

R=BASE/'runs/episode_000030_newton_observed_brush_20261021';REF='track3_hand_guided_brush_candidate'


def smooth(x):
    x=float(np.clip(x,0,1));return x*x*(3-2*x)


def main():
    assert not (R/'coherent-input-checksums.json').exists()
    path=next((R/'human_motion_data'/REF).rglob('motion.parquet'));backup=R/'research/before-handle-grip.parquet'
    assert not backup.exists();shutil.copy2(path,backup)
    row=pq.ParquetFile(path).read().to_pylist()[0];T=len(row['object_body_position'])
    fit=json.loads((R/'evaluation/refined-grasp-search-source-34-handle.json').read_text())['best_candidates']['mini_sweeper']
    assert fit['stable_hold_and_lift']
    local=np.array(fit['wrist_in_object']);fitted=np.array(fit['finger_joints'])
    w=np.array(row['robot_right_wrist_position']);q=np.array(row['robot_right_wrist_wxyz']);f=np.array(row['robot_right_finger_joints'])
    original_w=w.copy();original_q=q.copy();original_f=f.copy()
    for i in range(26,254):
        O=np.eye(4);O[:3,3]=row['object_body_position'][i][0];O[:3,:3]=Rotation.from_quat(row['object_body_wxyz'][i][0],scalar_first=True).as_matrix();hand=O@local
        weight=smooth((i-26)/8)*smooth((253-i)/8)
        w[i]=(1-weight)*original_w[i]+weight*hand[:3,3]
        if i<34:w[i,2]+=.04*np.sin(np.pi*(i-26)/8)
        q[i]=Slerp([0,1],Rotation.from_matrix([Rotation.from_quat(original_q[i],scalar_first=True).as_matrix(),hand[:3,:3]]))([weight]).as_quat(scalar_first=True)[0]
        closure=smooth((i-31)/3)*smooth((250-i)/5)
        f[i]=(1-closure)*original_f[i]+closure*fitted
    row['robot_right_wrist_position']=w.tolist();row['robot_right_wrist_wxyz']=q.tolist();row['robot_right_finger_joints']=f.tolist()
    pq.write_table(pa.Table.from_pylist([row]),path,compression='zstd')
    image=_docker(['inspect','v2d-batch-episode32-retarget','--format','{{.Image}}']).stdout.strip()
    shutil.copy2(BASE/'apply_collision_transitions.py',R/'scripts/apply_collision_transitions.py')
    _docker(['run','--rm','--name','v2d-brush-handle30-fk-cpu','--ipc','host','-e','CUDA_VISIBLE_DEVICES=',
             '-e',f'V2D_TRANSITION_REFERENCE={REF}','-e','V2D_REBUILD_ONLY=1','-e','V2D_CONTACT_APPLY_OUTPUT=research/brush-handle-fk.json',
             '-v',f'{R}:/episode','-v',f'{BASE}/video_to_data/robotic_grounding/source/robotic_grounding/robotic_grounding/assets:/v2d/assets:ro',
             '--entrypoint','/isaac-sim/python.sh',image,'-u','/episode/scripts/apply_collision_transitions.py'],R/'logs/brush-handle-fk.log')
    _docker(['exec','v2d-precision-episode30-newton','env','CUDA_VISIBLE_DEVICES=','JAX_PLATFORMS=cpu',f'V2D_AUDIT_REFERENCE={REF}',
             'python','-u','/episode/scripts/run_geometry_cpu.py','/episode/scripts/native_reference_audit.py'],R/'logs/brush-handle-collision-audit.log')
    for frame in [34,89,200]:
        docker(['exec','v2d-precision-episode30-newton','env',f'V2D_STATIC_GRASP_SOURCE_FRAME={frame}',
                'V2D_WRIST_MOTOR_PROFILE={"kp":250,"kd":20}','V2D_WRIST_SIDE_MOTOR_PROFILES={"left":{"kp":25,"kd":6}}',
                'V2D_STATIC_GRASP_LIFT_SIDES=["right"]','python','-u','/episode/scripts/diagnose_static_grasp.py'],R/f'logs/brush-handle-hold-lift-{frame}.log')
    save(R/'research/brush-handle-grip-provenance.json',{'updated_utc':utc(),'reference':REF,
         'held_source_interval':[34,245],'physical_grasp_fit':'evaluation/refined-grasp-search-source-34-handle.json',
         'source_hand_evidence':'Video/stereo human wrist is above and in front of the brush handle; prior robot reference gripped the lower body from the opposite side.',
         'object_poses_unchanged_by_grip_retargeting':True,'source_clock_unchanged':True,'object_assistance':0.})
    print('Handle grip rebuilt and physically checked at pickup, pause and brushing.',flush=True)


if __name__=='__main__':main()
