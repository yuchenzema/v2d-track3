"""Use the source-scored, restart-tested brush support pose before pickup."""
import json,shutil,subprocess
import numpy as np,pyarrow as pa,pyarrow.parquet as pq
from scipy.spatial.transform import Rotation,Slerp
from improvement_common import BASE,save,utc
from batch_native import _docker,docker

R=BASE/'runs/episode_000030_newton_observed_brush_20261021';REF='track3_stable_brush_pickup_candidate'


def smooth(x):
    x=float(np.clip(x,0,1));return x*x*(3-2*x)


def main():
    target=R/'human_motion_data'/REF;assert not target.exists()
    search=json.loads((R/'research/initial-brush-support-search.json').read_text());best=search['best'];assert best['stable_and_plausible']
    assert best['restart_maximum_translation_m']<=.01 and best['restart_maximum_rotation_rad']<=.2
    shutil.copytree(R/'human_motion_data/track3_ray_projected_brush_candidate',target)
    path=next(target.rglob('motion.parquet'));row=pq.ParquetFile(path).read().to_pylist()[0];T=len(row['object_body_position'])
    O=np.tile(np.eye(4),(T,2,1,1));O[:,:,:3,3]=row['object_body_position'];O[:,:,:3,:3]=Rotation.from_quat(np.array(row['object_body_wxyz']).reshape(-1,4),scalar_first=True).as_matrix().reshape(T,2,3,3);old=O.copy()
    stable=np.eye(4);stable[:3,3]=best['settled_pose'][:3];stable[:3,:3]=Rotation.from_quat(best['settled_pose'][3:]).as_matrix();local=np.linalg.inv(O[0,1])@stable
    for i in range(65):
        A=O[i,1]@local;weight=smooth((i-50)/15)
        O[i,0,:3,3]=(1-weight)*A[:3,3]+weight*old[i,0,:3,3]
        O[i,0,:3,:3]=Slerp([0,1],Rotation.from_matrix([A[:3,:3],old[i,0,:3,:3]]))([weight])[0].as_matrix()
    w=np.array(row['robot_right_wrist_position']);q=np.array(row['robot_right_wrist_wxyz'])
    for i in range(26,65):
        delta=O[i,0]@np.linalg.inv(old[i,0]);weight=smooth((i-26)/8)
        w[i]=(1-weight)*w[i]+weight*(delta[:3,:3]@w[i]+delta[:3,3])
        q[i]=(Rotation.from_rotvec(Rotation.from_matrix(delta[:3,:3]).as_rotvec()*weight)*Rotation.from_quat(q[i],scalar_first=True)).as_quat(scalar_first=True)
    row['object_body_position']=O[:,:,:3,3].tolist();row['object_body_wxyz']=Rotation.from_matrix(O[:,:,:3,:3].reshape(-1,3,3)).as_quat(scalar_first=True).reshape(T,2,4).tolist()
    row['robot_right_wrist_position']=w.tolist();row['robot_right_wrist_wxyz']=q.tolist();pq.write_table(pa.Table.from_pylist([row]),path,compression='zstd')
    image=_docker(['inspect','v2d-batch-episode32-retarget','--format','{{.Image}}']).stdout.strip()
    _docker(['run','--rm','--name','v2d-stable-brush30-fk-cpu','--ipc','host','-e','CUDA_VISIBLE_DEVICES=',
        '-e',f'V2D_TRANSITION_REFERENCE={REF}','-e','V2D_REBUILD_ONLY=1','-e','V2D_CONTACT_APPLY_OUTPUT=research/stable-brush-fk.json',
        '-v',f'{R}:/episode','-v',f'{BASE}/video_to_data/robotic_grounding/source/robotic_grounding/robotic_grounding/assets:/v2d/assets:ro',
        '--entrypoint','/isaac-sim/python.sh',image,'-u','/episode/scripts/apply_collision_transitions.py'],R/'logs/stable-brush-fk.log')
    docker(['exec','v2d-precision-episode30-newton','env',f'V2D_DIAGNOSTIC_REFERENCE={REF}','V2D_DIAGNOSTIC_SOURCE_FRAME=0',
        'V2D_DIAGNOSTIC_LABEL=stable-brush-initial-rest','V2D_DIAGNOSTIC_SUBSTEPS=20','python','-u','/episode/scripts/diagnose_terminal_support.py'],R/'logs/stable-brush-initial-rest.log')
    rest=json.loads((R/'evaluation/stable-brush-initial-rest.json').read_text());assert all(v['passed_1cm_0_2rad_rest_test'] for v in rest['objects']),rest
    with (R/'logs/stable-brush-source-fidelity.log').open('w') as log:
        subprocess.run([str(R/'.venv-video/bin/python'),str(BASE/'assess_physical_reference.py'),'--run',str(R),'--reference',REF,'--output','stable-brush-source-fidelity','--no-update-selection'],stdout=log,stderr=subprocess.STDOUT,check=True)
    save(R/'research/stable-brush-pickup-reference.json',{'updated_utc':utc(),'reference':REF,'approved_for_independent_training':True,
         'source_frames':T,'initial_support_search':'research/initial-brush-support-search.json','independent_rest_check':rest,
         'stationary_holder_interval':[0,50],'extraction_transition':[50,65],'poses_after_source65_unchanged':True,
         'existing_training_reference_unchanged':True,'object_assistance':0.})
    print('Stable brush pickup reference passed independent initial-rest and full-source image checks.',flush=True)


if __name__=='__main__':main()
