"""Open fingers before wrist withdrawal; object trajectories remain video-derived."""
import argparse,json,subprocess
from pathlib import Path
import numpy as np,pyarrow as pa,pyarrow.parquet as pq
from scipy.spatial.transform import Rotation,Slerp
from improvement_common import save,utc
p=argparse.ArgumentParser();p.add_argument('--run',type=Path,required=True);a=p.parse_args();r=a.run;c=json.loads((r/'episode-config.json').read_text());marker=r/'research/rebuilt-release-targets.json'
overlay=r/'videos/reference-source-fidelity-overlay.mp4';preview=r/'videos/reconstruction-source-review.mp4'
if overlay.exists() and not preview.exists():subprocess.run(['ffmpeg','-hide_banner','-loglevel','error','-y','-i',str(overlay),'-c:v','libx264','-threads','2','-preset','fast','-crf','21','-pix_fmt','yuv420p','-movflags','+faststart',str(preview)],check=True)
subprocess.run([str(r/'.venv-video/bin/python'),str(r.parents[1]/'rebuild_full_physics_preflight_20261022.py'),'--run',str(r)],check=True)
if marker.exists():raise SystemExit(0)
path=next((r/'human_motion_data/track3_collision_variant').rglob('motion.parquet'));row=pq.ParquetFile(path).read().to_pylist()[0];T=c['frames'];changes=[]
def smooth(x):x=float(np.clip(x,0,1));return x*x*(3-2*x)
for side in ['left','right']:
 w=np.asarray(row[f'robot_{side}_wrist_position']);q=np.asarray(row[f'robot_{side}_wrist_wxyz']);f=np.asarray(row[f'robot_{side}_finger_joints']);initial_open=np.asarray(row[f'robot_{side}_finger_joints'][0]).copy()
 if c['episode']==20 and side=='left':
  # This hand is already holding the cup in source frame zero; use its observed end-of-video open posture.
  initial_open=f[-1].copy()
 for g in [g for g in c['grasps'] if g['side']==side]:
  end=g['end']
  if end>=T-4:continue
  held_w=w[end].copy();held_q=q[end].copy();held_f=f[end].copy();last=min(T-1,end+10);destination=w[last].copy();destination[2]=max(destination[2],held_w[2]+.035)
  for i in range(end+1,last+1):
   opening=smooth((i-end)/3);f[i]=(1-opening)*held_f+opening*initial_open
   if i<=end+3:w[i]=held_w;q[i]=held_q
   else:
    lift=smooth((i-end-3)/3);out=smooth((i-end-6)/max(1,last-end-6));w[i]=(1-out)*(held_w+[0,0,.035*lift])+out*destination;q[i]=Slerp([0,1],Rotation.from_quat([held_q,q[last]],scalar_first=True))([out]).as_quat(scalar_first=True)[0]
  changes.append({'side':side,'object':c['objects'][g['object']],'observed_release_source_frame':end,'finger_opening_source_interval':[end+1,min(T-1,end+3)],'wrist_clearance_interval':[end+4,last],'object_pose_columns_modified':False})
 for key,value in [('wrist_position',w),('wrist_wxyz',q),('finger_joints',f)]:row[f'robot_{side}_{key}']=value.tolist()
pq.write_table(pa.Table.from_pylist([row]),path,compression='zstd');save(marker,{'updated_utc':utc(),'changes':changes,'source_time_grid_preserved':True,'final_finger_postures_are_native_robot_joint_targets':True})
