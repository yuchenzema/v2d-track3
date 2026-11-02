"""Re-anchor a physically captured grasp to the unchanged source object pose.

The resulting reference is diagnostic only: FK/contact labels must be rebuilt
and the complete motion validated before it can become a training reference.
"""
from pathlib import Path
import json
import shutil
import xml.etree.ElementTree as ET
import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq
from scipy.spatial.transform import Rotation,Slerp
from flash_chord.data import load_reference
from flash_chord.embodiments.sharpa_hands import SharpaHandsConfig

ROOT=Path('/episode');cfg=json.loads((ROOT/'episode-config.json').read_text())
source=ROOT/'human_motion_data'/cfg['training_reference']
path=next(source.rglob('motion.parquet'));row=pq.ParquetFile(path).read().to_pylist()[0]
reference=load_reference(path.parent,control_fps=20.,motion_speed=.5)
grid=json.loads((ROOT/'evaluation/grasp-offset-grid.json').read_text())
data=np.load(ROOT/'evaluation/grasp-offset-grid.npz');poses=data['body_poses']
worlds=len(grid['profiles']);bodies=poses.shape[1]//worlds;names=data['body_names'].tolist()
frame=int(round(grid['source_frame']*(reference.num_frames-1)/(cfg['frames']-1)))
objects=np.asarray(row['object_body_position']);quats=np.asarray(row['object_body_wxyz'])
fits=[];rejections=[]


def matrix(pose):
    value=np.eye(4);value[:3,3]=pose[:3];value[:3,:3]=Rotation.from_quat(pose[3:]).as_matrix();return value


for body,name in enumerate(cfg['objects']):
    grasp=next(g for g in cfg['grasps'] if g['object']==body and g['start']<=grid['source_frame']<=g['end'])
    side=grasp['side'];palm=names[:bodies].index(f'{side}_sharpa_wave/{side}_hand_C_MC')
    eligible=[]
    for record in grid['profiles']:
        if record['profile']['side'] not in [side,'both']:continue
        world=record['world'];oi=int(data['object_body_ids'][world,body]);wi=palm+world*bodies
        obj=poses[:,oi];hand=poses[:,wi]
        lift=obj[-1,:3]-obj[19,:3];hand_lift=hand[-1,:3]-hand[19,:3]
        error=float(np.linalg.norm(lift-hand_lift))
        angle=float((Rotation.from_quat(obj[19,3:]).inv()*Rotation.from_quat(obj[-1,3:])).magnitude())
        if lift[2]<.015 or hand_lift[2]<.02 or error>.01 or angle>.15:continue
        eligible.append((error+.02*angle+.1*record['objects'][name]['static_position_drift_m'],record))
    if not eligible:
        rejections.append({'object':name,'reason':'No candidate physically carried this object during the test lift'});continue
    _,record=min(eligible,key=lambda item:item[0]);world=record['world'];profile=record['profile']
    direction=np.asarray(grid['grasp_directions'][side]['world_inward_direction'])
    command=np.eye(4);command[:3,:3]=Rotation.from_quat(reference.wrist_quat_w(side)[frame],scalar_first=True).as_matrix()
    command[:3,3]=reference.wrist_pos_w(side)[frame]+direction*profile['inward_m']+[0,0,.03]
    oi=int(data['object_body_ids'][world,body]);local=np.linalg.inv(matrix(poses[-1,oi]))@command
    finger=np.asarray(reference.finger_joint_pos(side)[frame]).copy();joint_names=reference.finger_joint_names(side)
    urdf=getattr(SharpaHandsConfig(),side+'_urdf_path');limits={j.attrib['name']:j.find('limit').attrib for j in ET.parse(urdf).getroot().findall('joint') if j.find('limit') is not None}
    for j,joint in enumerate(joint_names):
        if joint.endswith(('_FE','_IP','_PIP','_DIP')):finger[j]+=profile['finger_rad']
        limit=limits.get(joint) or limits.get(joint.removeprefix(side+'_'))
        if limit is None:raise ValueError(f'Missing actual robot joint limits for {joint}')
        finger[j]=np.clip(finger[j],float(limit['lower']),float(limit['upper']))
    fits.append({'side':side,'object_index':body,'object':name,'held_interval':[grasp['start'],grasp['end']],
        'template_source_frame':grid['source_frame'],'wrist_in_object':local.tolist(),'finger_joints':finger.tolist(),
        'derived_from_physical_world':world,'physical_profile':profile,
        'original_static_drift_m':record['objects'][name]['static_position_drift_m']})

before_position=np.asarray(row['object_body_position']).copy();before_quat=np.asarray(row['object_body_wxyz']).copy()
for fit in fits:
    side=fit['side'];body=fit['object_index'];start,end=fit['held_interval']
    w=np.asarray(row[f'robot_{side}_wrist_position']);q=np.asarray(row[f'robot_{side}_wrist_wxyz']);f=np.asarray(row[f'robot_{side}_finger_joints'])
    for t in range(max(0,start-5),min(cfg['frames'],end+6)):
        weight=float(np.clip(min((t-start+5)/5.,(end+5-t)/5.),0.,1.))
        weight=weight*weight*(3-2*weight)
        obj=np.eye(4);obj[:3,3]=objects[t,body];obj[:3,:3]=Rotation.from_quat(quats[t,body],scalar_first=True).as_matrix()
        target=obj@np.asarray(fit['wrist_in_object']);w[t]=(1-weight)*w[t]+weight*target[:3,3]
        q[t]=Slerp([0,1],Rotation.from_matrix([Rotation.from_quat(q[t],scalar_first=True).as_matrix(),target[:3,:3]]))([weight]).as_quat(scalar_first=True)[0]
        f[t]=(1-weight)*f[t]+weight*np.asarray(fit['finger_joints'])
    for key,value in [('wrist_position',w),('wrist_wxyz',q),('finger_joints',f)]:row[f'robot_{side}_{key}']=value.tolist()
assert np.array_equal(before_position,row['object_body_position']) and np.array_equal(before_quat,row['object_body_wxyz'])
destination=ROOT/'human_motion_data/track3_settled_grasp_probe'
if not destination.exists():shutil.copytree(source,destination)
output=next(destination.rglob('motion.parquet'));pq.write_table(pa.Table.from_pylist([row]),output,compression='zstd')
report={'episode':cfg['episode'],'diagnostic_reference':'track3_settled_grasp_probe',
    'approved_for_training':False,'forward_kinematics_and_contact_labels_pending':True,
    'source_object_poses_unchanged':True,'source_camera_unchanged':True,'grasps':fits,'rejections':rejections}
(ROOT/'research/settled-grasp-probe-reference.json').write_text(json.dumps(report,indent=2))
print(json.dumps(report,indent=2),flush=True)
