"""Fix camera gauge using robust static-table anchors and metric SLAM motion.

Per-frame circle fitting is used as a measurement and outlier diagnostic, not
as independent per-frame camera translation. This avoids occlusion-induced
camera jumps being imposed on both hands and manipulated objects.
"""
from pathlib import Path
import json,shutil
import numpy as np
from scipy.spatial.transform import Rotation

def stabilize(root):
 root=Path(root);rec=root/'reconstruction';marker=rec/'camera-stability.json'
 if marker.exists():return False
 cfg=json.loads((root/'episode-config.json').read_text())
 if cfg.get('camera_odometry_method')=='sift_pnp_metric':
  from metric_table_anchor import run
  run(root)
  marker.write_text((rec/'metric-table-anchor.json').read_text())
  return True
 cfg=json.loads((root/'episode-config.json').read_text());d=np.load(rec/'table-camera-calibration.npz');old={k:d[k].copy() for k in d.files};S=[]
 for i in range(cfg['frames']):
  p=json.loads((rec/f'slam/poses/{i:06d}.json').read_text());M=np.eye(4);M[:3,:3]=Rotation.from_quat(p['rotation'],scalar_first=True).as_matrix();M[:3,3]=p['translation'];S.append(M)
 S=np.array(S);planes=old['table_planes_camera'];radii=old['measured_radii'];scores=old['circle_inlier_fraction'];radius=float(np.median(radii));good=(np.abs(radii-radius)<.012)&(scores>.2)
 if good.sum()<12:good=np.abs(radii-radius)<.02
 assert good.sum()>=8
 anchors=np.einsum('nij,nj->ni',S[:,:3,:3],old['table_centers_camera'])+S[:,:3,3];center=np.median(anchors[good],axis=0);up=np.median(np.einsum('nij,nj->ni',S[:,:3,:3],planes[:,:3])[good],axis=0);up/=np.linalg.norm(up);x=S[0,:3,0].copy();x-=up*(x@up);x/=np.linalg.norm(x);world=np.eye(4);world[:3,:3]=np.stack([x,np.cross(up,x),up]);world[:3,3]=np.array([0.,0.,.65])-world[:3,:3]@center;camera=world@S
 assert np.isfinite(camera).all()
 original=old['camera_to_world'];delta_old=np.linalg.norm(np.diff(original[:,:3,3],axis=0),axis=1);delta_new=np.linalg.norm(np.diff(camera[:,:3,3],axis=0),axis=1);residual=np.linalg.norm(anchors-center,axis=1)
 report={'method':__doc__,'source':'Metric DROID SLAM, aligned to calibrated stereo depth; static tabletop gives only robust global orientation, metric radius and gauge origin','independent_per_frame_circle_translation_removed':True,'retained_table_anchor_frames':np.flatnonzero(good).tolist(),'median_table_anchor_residual_m':float(np.median(residual[good])),'p95_table_anchor_residual_m':float(np.quantile(residual[good],.95)),'original_camera_step_m_p50_p95_max':np.quantile(delta_old,[.5,.95,1]).tolist(),'stabilized_camera_step_m_p50_p95_max':np.quantile(delta_new,[.5,.95,1]).tolist(),'table_radius_m':radius,'frame_count':len(S),'evaluation_ground_truth_used':False,'caveat':'Metric SLAM and visible plane remain estimates; scene gauge is chosen, not a released mocap calibration.'}
 backup=root/'research/superseded_camera';backup.mkdir(exist_ok=True)
 for name in ['table-camera-calibration.npz','table-camera-calibration.json','scene_motion.npz','scene_motion_raw.npz','scene_motion_physical.npz','physical-refinement.json','scene-reconstruction.json','rebuilt_scene.glb','tabletop_video_texture.png']:
  p=rec/name
  if p.exists():shutil.move(p,backup/name)
 for name in ['reference-source-fidelity.json','reference-source-fidelity-review-00.jpg','reference-source-fidelity-review-04.jpg','reference-source-fidelity-review-08.jpg']:
  p=root/'evaluation'/name
  if p.exists():shutil.move(p,backup/name)
 for name in ['track3_video','track3_coherent_variant']:
  p=root/'human_motion_data'/name
  if p.exists():
   destination=backup/name
   if destination.exists():
    import time
    destination=backup/(name+'-copy-'+str(time.time_ns()))
   shutil.move(p,destination)
 # Keep raw plane/circle measurements unchanged for reproducibility.
 old['camera_to_world']=camera;np.savez_compressed(rec/'table-camera-calibration.npz',**old)
 previous=json.loads((backup/'table-camera-calibration.json').read_text());previous.update(method=__doc__,translation_source='Metric SLAM relative translation; robust static-table global gauge, no per-frame circle jumps',orientation_source='Metric SLAM relative rotation; robust gravity inferred from table planes');(rec/'table-camera-calibration.json').write_text(json.dumps(previous,indent=2));marker.write_text(json.dumps(report,indent=2))
 cfg['reconstruction_visual_reviewed']=False;(root/'episode-config.json').write_text(json.dumps(cfg,indent=2));print('Stabilized',cfg['sequence'],json.dumps({k:v for k,v in report.items() if k.endswith('_max') or k.endswith('_residual_m')}),flush=True);return True
if __name__=='__main__':
 import sys
 stabilize(sys.argv[1])
