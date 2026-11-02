"""Combine the visible stationary pickup with a hand-guided rigid brush fit."""
import json,shutil
import numpy as np,pyarrow as pa,pyarrow.parquet as pq
from scipy.spatial.transform import Rotation,Slerp
import prepare_observed_brush_reference as probe
from improvement_common import save,utc

R=probe.R;REF='track3_hand_guided_brush_candidate'


def blend(A,B,weight):
    C=np.eye(4);C[:3,3]=(1-weight)*A[:3,3]+weight*B[:3,3]
    C[:3,:3]=Slerp([0,1],Rotation.from_matrix([A[:3,:3],B[:3,:3]]))([weight])[0].as_matrix()
    return C


def build():
    target=R/'human_motion_data'/REF;assert not target.exists()
    shutil.copytree(R/'human_motion_data/track3_observed_brush_pickup_candidate',target)
    path=next(target.rglob('motion.parquet'));row=pq.ParquetFile(path).read().to_pylist()[0]
    P=np.array(row['object_body_position']);Q=np.array(row['object_body_wxyz']);T=len(P)
    O=np.tile(np.eye(4),(T,2,1,1));O[:,:,:3,3]=P;O[:,:,:3,:3]=Rotation.from_quat(Q.reshape(-1,4),scalar_first=True).as_matrix().reshape(T,2,3,3)
    original=O.copy();guided=np.load(R/'reconstruction/scene_motion_brush_hand_guided_candidate.npz')['object_to_world'][:,0]
    anchor=np.linalg.inv(O[50,1])@O[50,0]
    for frame in range(51,246):
        if frame<65:O[frame,0]=blend(O[frame,1]@anchor,guided[frame],probe.smooth((frame-50)/15))
        elif frame<235:O[frame,0]=guided[frame]
        else:O[frame,0]=blend(guided[frame],original[frame,0],probe.smooth((frame-235)/10))
    early=json.loads((R/'evaluation/refined-grasp-search-source-34.json').read_text())['best_candidates']['mini_sweeper']
    late=json.loads((R/'evaluation/reanchored-grasp-search.json').read_text())['best_candidates']['mini_sweeper']
    A=np.array(early['wrist_in_object']);B=np.array(late['wrist_in_object'])
    w=np.array(row['robot_right_wrist_position']);q=np.array(row['robot_right_wrist_wxyz']);f=np.array(row['robot_right_finger_joints'])
    for frame in range(34,246):
        weight=probe.smooth((frame-54)/11);local=blend(A,B,weight);hand=O[frame,0]@local
        w[frame]=hand[:3,3];q[frame]=Rotation.from_matrix(hand[:3,:3]).as_quat(scalar_first=True)
        f[frame]=(1-weight)*np.array(early['finger_joints'])+weight*np.array(late['finger_joints'])
    row['object_body_position']=O[:,:,:3,3].tolist();row['object_body_wxyz']=Rotation.from_matrix(O[:,:,:3,:3].reshape(-1,3,3)).as_quat(scalar_first=True).reshape(T,2,4).tolist()
    row['robot_right_wrist_position']=w.tolist();row['robot_right_wrist_wxyz']=q.tolist();row['robot_right_finger_joints']=f.tolist()
    pq.write_table(pa.Table.from_pylist([row]),path,compression='zstd')
    save(R/'research/hand-guided-brush-reference.json',{'updated_utc':utc(),'reference':REF,'approved_for_training':False,
        'source_frames':T,'diagnostic_only':True,'hand_guided_object_reconstruction':'reconstruction/brush-hand-guided-reconstruction.json',
        'held_source_mask_iou_before':.6714831617844915,'held_source_mask_iou_before_final_smoothing':.800942801756865,
        'stationary_holder_interval':[0,50],'visible_extraction_blend':[50,65],'terminal_blend':[235,245],
        'rack_camera_visual_geometry_and_source_clock_unchanged':True})
    probe.REF=REF;probe.review_candidate()


if __name__=='__main__':build()
