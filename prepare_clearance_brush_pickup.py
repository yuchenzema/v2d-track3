"""Match the observed lift-before-sideways extraction from the rack pegs."""
import json,shutil
import numpy as np,pyarrow as pa,pyarrow.parquet as pq
from scipy.interpolate import PchipInterpolator
from scipy.spatial.transform import Rotation,Slerp
from improvement_common import BASE,save,utc
from batch_native import _docker,docker

R=BASE/'runs/episode_000030_newton_stable_brush_20261021';REF='track3_clearance_brush_pickup_candidate'


def main():
    target=R/'human_motion_data'/REF;assert not target.exists()
    shutil.copytree(R/'human_motion_data/track3_stable_brush_pickup_candidate',target)
    path=next(target.rglob('motion.parquet'));row=pq.ParquetFile(path).read().to_pylist()[0]
    P=np.array(row['object_body_position']);Q=np.array(row['object_body_wxyz']);beforeP=P.copy();beforeQ=Q.copy()
    a=P[50,0].copy();b=P[65,0].copy();high=max(a[2]+.05,b[2]+.02)
    keys=np.array([a,a+[0,0,.05],(a+b)/2,b]);keys[2,2]=high
    curve=PchipInterpolator([50,55,59,65],keys,axis=0)
    for i in range(50,66):
        P[i,0]=curve(i);weight=float(np.clip((i-55)/10,0,1));weight=weight*weight*(3-2*weight)
        Q[i,0]=Slerp([0,1],Rotation.from_quat([beforeQ[50,0],beforeQ[65,0]],scalar_first=True))([weight]).as_quat(scalar_first=True)[0]
    w=np.array(row['robot_right_wrist_position']);q=np.array(row['robot_right_wrist_wxyz'])
    for i in range(50,66):
        A=Rotation.from_quat(beforeQ[i,0],scalar_first=True);B=Rotation.from_quat(Q[i,0],scalar_first=True);delta=B*A.inv()
        w[i]=delta.apply(w[i]-beforeP[i,0])+P[i,0];q[i]=(delta*Rotation.from_quat(q[i],scalar_first=True)).as_quat(scalar_first=True)
    row['object_body_position']=P.tolist();row['object_body_wxyz']=Q.tolist();row['robot_right_wrist_position']=w.tolist();row['robot_right_wrist_wxyz']=q.tolist()
    pq.write_table(pa.Table.from_pylist([row]),path,compression='zstd')
    image=_docker(['inspect','v2d-batch-episode32-retarget','--format','{{.Image}}']).stdout.strip()
    _docker(['run','--rm','--name','v2d-clearance-brush30-fk-cpu','--ipc','host','-e','CUDA_VISIBLE_DEVICES=',
        '-e',f'V2D_TRANSITION_REFERENCE={REF}','-e','V2D_REBUILD_ONLY=1','-e','V2D_CONTACT_APPLY_OUTPUT=research/clearance-brush-fk.json',
        '-v',f'{R}:/episode','-v',f'{BASE}/video_to_data/robotic_grounding/source/robotic_grounding/robotic_grounding/assets:/v2d/assets:ro',
        '--entrypoint','/isaac-sim/python.sh',image,'-u','/episode/scripts/apply_collision_transitions.py'],R/'logs/clearance-brush-fk.log')
    cfg=json.loads((R/'episode-config.json').read_text());container=cfg['native_container']
    for name in ['wrist_motor_controller.py','diagnose_reconstruction_motion.py','diagnose_zero_residual.py']:
        if (BASE/name).exists():shutil.copy2(BASE/name,R/'scripts'/name)
    label='clearance-brush-untrained-feasibility'
    docker(['exec',container,'env',f'V2D_DIAGNOSTIC_REFERENCE={REF}','V2D_DIAGNOSTIC_SUBSTEPS=20',f'V2D_DIAGNOSTIC_LABEL={label}',
        'V2D_WRIST_MOTOR_PROFILE={"kp":250,"kd":20}','V2D_WRIST_SIDE_MOTOR_PROFILES={"left":{"kp":25,"kd":6}}',
        'python','-u','/episode/scripts/diagnose_reconstruction_motion.py'],R/f'logs/{label}.log')
    save(R/'research/clearance-brush-pickup-reference.json',{'updated_utc':utc(),'reference':REF,'diagnostic_only':True,
         'source_evidence':'Frames50..55 lift the brush free of the rack before the sideways motion at58..65.',
         'modified_source_frames':[50,65],'lift_height_m':.05,'initial_support_unchanged':True,
         'camera_CAD_and_source_clock_unchanged':True,'object_assistance':0.,'policy_improvement_verified':False})
    print('Lift-before-sideways brush extraction reference tested with untrained physical hand control.',flush=True)


if __name__=='__main__':main()
