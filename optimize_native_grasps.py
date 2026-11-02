"""Fit grasp postures against actual Newton/MuJoCo link collisions."""
import os,json,runpy,sys,numpy as np,mujoco,xml.etree.ElementTree as ET
from pathlib import Path
import pyarrow.parquet as pq
from scipy.spatial.transform import Rotation
from scipy.optimize import least_squares
from rigid_pose_collision_audit import contact_retry_thresholds
import flash_chord.training.flash_sac.runner as runner
R=Path('/episode');cfg=json.loads((R/'episode-config.json').read_text());seq=cfg['sequence'];reference=os.environ.get('V2D_GRASP_REFERENCE','track3_coherent_variant');frame_mode=os.environ.get('V2D_OPTIMIZE_CONTACT_FRAMES','0')=='1';source_sigma=float(os.environ.get('V2D_SOURCE_HAND_SIGMA','0'));ref=R/f'human_motion_data/{reference}/processed/sequence_id={seq}/robot_name=sharpa_wave/motion.parquet';c=pq.ParquetFile(ref).read().to_pylist()[0];templates=[] if frame_mode else json.loads((R/'research/coherent-grasp-variant.json').read_text())
strict_contacts=cfg.get('require_nonpenetrating_reference_before_training',False)
selection_mm=.001 if strict_contacts else 3.
contact_allowance_m=-.0005 if strict_contacts else .0005
def inspect(env,*args,**kwargs):
 e=env.env;solver=e.solver;m=solver.mj_model;d=solver.mj_data;names={kind:{mujoco.mj_id2name(m,kind,i):i for i in range(count)} for kind,count in [(mujoco.mjtObj.mjOBJ_BODY,m.nbody),(mujoco.mjtObj.mjOBJ_JOINT,m.njnt)]};results=[];O=np.array(c['object_body_position']);OQ=np.array(c['object_body_wxyz'])
 def body(n):return names[mujoco.mjtObj.mjOBJ_BODY][n]
 def joint(n):return names[mujoco.mjtObj.mjOBJ_JOINT][n]
 jobs=json.loads(os.environ['V2D_GRASP_JOBS']) if os.environ.get('V2D_GRASP_JOBS') else cfg['grasps']
 if frame_mode:
  audit=json.loads((R/f'evaluation/{reference}-collision-audit.json').read_text());jobs=[]
  for row in audit['frames']:
   for side in ['left','right']:
    frame_selection=json.loads(os.environ['V2D_CONTACT_FRAME_SELECTION']) if os.environ.get('V2D_CONTACT_FRAME_SELECTION') else None
    if frame_selection is not None and row['source_frame'] not in frame_selection.get(side,[]):continue
    hits=[x for x in row['penetrating_cross_body_contacts'] if x['penetration_mm']>selection_mm and any(n.startswith(side+'_sharpa_wave_') for n in x['bodies'])]
    if not hits:continue
    touched=[b for b,n in enumerate(cfg['objects']) if any(any('track3_'+n in body for body in x['bodies']) for x in hits)]
    b=touched[0] if touched else next((g['object'] for g in cfg['grasps'] if g['side']==side),0);t=row['source_frame'];jobs.append({'side':side,'object':b,'start':t,'end':t,'template_source_frame':t})
 for gi,g in enumerate(jobs):
  side=g['side'];b=g['object'];template=g if frame_mode or 'template_source_frame' in g else (templates[side] if isinstance(templates,dict) else next(t for t in templates if t['side']==side and t['held_interval']==[g['start'],g['end']]));t=template['template_source_frame'];frame=int(round(t*(e.reference.num_frames-1)/(cfg['frames']-1)));e.reset(frame_id=frame);solver._update_mjc_data(d,e.model,e.state_0);mujoco.mj_forward(m,d)
  prefix=f'{side}_sharpa_wave_';wrist=body(prefix+f'{side}_hand_C_MC');adr=int(m.jnt_qposadr[joint(f'{side}_wrist_x')]);assert int(m.jnt_qposadr[joint(f'{side}_wrist_rotvec')])==adr+3;q0=d.qpos.copy();w=q0[adr:adr+3].copy();rot=Rotation.from_quat(q0[adr+3:adr+7],scalar_first=True)
  assert np.linalg.norm(w-np.array(c[f'robot_{side}_wrist_position'])[t])<1e-5
  joint_names=c[f'{side}_robot_finger_joint_names'];joints=[joint(prefix+n) for n in joint_names];indices=np.array([m.jnt_qposadr[j] for j in joints]);finger=q0[indices].copy();assert np.max(np.abs(finger-np.array(c[f'robot_{side}_finger_joints'])[t]))<1e-5
  handbodies={i for n,i in names[mujoco.mjtObj.mjOBJ_BODY].items() if n and n.startswith(prefix)};handgeoms=np.flatnonzero(np.isin(m.geom_bodyid,list(handbodies)));geom_indices={int(j):i for i,j in enumerate(handgeoms)}
  xml=ET.parse(Path('/v2d/assets/xmls/sharpawave')/f'{side}_sharpawave.xml');tips=[]
  for f in ['thumb','index','middle','ring','pinky']:
   bi=body(prefix+f'{side}_{f}_DP');node=xml.find(f'.//site[@name="{side}_{f}_tip_site"]');offset=np.fromstring(node.get('pos'),sep=' ');tips.append((bi,offset))
  def tip_positions():return np.array([d.xmat[bi].reshape(3,3)@offset+d.xpos[bi] for bi,offset in tips])
  if source_sigma:
   tracked=[(wrist,np.zeros(3))]
   for finger_index,finger_name in enumerate(['thumb','index','middle','ring','pinky']):
    segments=['CMC_VL','MCP_VL','DP'] if finger_name=='thumb' else ['MCP_VL','MP','DP'];tracked.extend((body(prefix+f'{side}_{finger_name}_{segment}'),np.zeros(3)) for segment in segments);tracked.append(tips[finger_index])
   camera=np.linalg.inv(np.load(R/'reconstruction/scene_motion_physical.npz')['camera_to_world'][t]);intr=json.loads((R/'reconstruction/camera_intrinsics.json').read_text());measurement=np.load(R/'reconstruction/hand_measurements.npz');side_index=['left','right'].index(side)
   if measurement['observed'][t,side_index]:source_uv=measurement['uv'][t,side_index].copy()
   else:
    points=np.load(R/'reconstruction/scene_motion_physical.npz')['hand_joint_positions'][t,side_index];pc=points@camera[:3,:3].T+camera[:3,3];source_uv=pc[:,:2]/pc[:,2:]*[intr['fx'],intr['fy']]+[intr['cx'],intr['cy']]
   def image_error():
    points=np.array([d.xmat[bi].reshape(3,3)@offset+d.xpos[bi] for bi,offset in tracked]);pc=points@camera[:3,:3].T+camera[:3,3];return (pc[:,:2]/np.maximum(pc[:,2:],.05)*[intr['fx'],intr['fy']]+[intr['cx'],intr['cy']]-source_uv)
  target=tip_positions();active=np.array(c[f'mano_{side}_object_contact_part_ids'])[t]>0
  if frame_mode and not any(g['side']==side and g['start']<=t<=g['end'] for g in cfg['grasps']):active[:]=False
  def update(x):
   d.qpos[:]=q0;d.qpos[adr:adr+3]=w+x[:3];d.qpos[adr+3:adr+7]=(Rotation.from_rotvec(x[3:6])*rot).as_quat(scalar_first=True);d.qpos[indices]=x[6:];mujoco.mj_forward(m,d)
  def collisions():
   if strict_contacts:
    from rigid_pose_collision_audit import hand_clearance_penalties
    return hand_clearance_penalties(m,d,handgeoms,handbodies)
   penalties=np.zeros(len(handgeoms));maximum=0.
   for contact in d.contact:
    a,b=map(int,contact.geom);ha=a in geom_indices;hb=b in geom_indices
    if ha==hb:continue
    depth=max(0.,-float(contact.dist));maximum=max(maximum,depth);gi=geom_indices[a if ha else b];penalties[gi]=max(penalties[gi],max(0.,-float(contact.dist)-contact_allowance_m))
   return penalties,maximum
  def residual(x):
   update(x);contact,_=collisions();tip_error=(tip_positions()-target)/np.where(active[:,None],.003,.035)
   image=image_error().ravel()/source_sigma if source_sigma else np.empty(0)
   return np.r_[contact/.00025,tip_error.ravel(),x[:3]/.025,x[3:6]/.4,(x[6:]-finger)/.5,image]
  translation_limit=.09 if strict_contacts else .04
  initial=np.r_[np.zeros(6),finger];_,before=collisions();lo=np.r_[np.full(3,-translation_limit),np.full(3,-.7),[m.jnt_range[j,0] if m.jnt_limited[j] else -3.14 for j in joints]];hi=np.r_[np.full(3,translation_limit),np.full(3,.7),[m.jnt_range[j,1] if m.jnt_limited[j] else 3.14 for j in joints]];initial=np.clip(initial,lo+1e-7,hi-1e-7)
  solution=least_squares(residual,initial,bounds=(lo,hi),max_nfev=300,diff_step=1e-4,ftol=1e-7,xtol=1e-7,gtol=1e-7,x_scale=np.r_[np.full(3,.01),np.full(3,.2),np.full(len(finger),.3)])
  update(solution.x);_,after=collisions()
  opening_threshold,escape_threshold=contact_retry_thresholds(strict_contacts)
  if after>opening_threshold:
   options=[solution]
   for opening in [.65,.35]:
    seed=initial.copy();seed[6:]=np.clip(finger*opening,lo[6:]+1e-7,hi[6:]-1e-7)
    candidate=least_squares(residual,seed,bounds=(lo,hi),max_nfev=350,diff_step=1e-4,ftol=1e-7,xtol=1e-7,gtol=1e-7,x_scale=np.r_[np.full(3,.01),np.full(3,.2),np.full(len(finger),.3)]);options.append(candidate)
   def merit(candidate):
    update(candidate.x);_,depth=collisions();return (max(0.,depth-(.000001 if strict_contacts else .0015)),float(np.sum(residual(candidate.x)**2)))
   solution=min(options,key=merit);update(solution.x);_,after=collisions()
  if after>escape_threshold:
   # Deep palm/rim contact has local minima that opening the fingers alone
   # cannot leave. Seed a bounded retraction and an upward escape explicitly.
   choices=[solution];center=np.array([*O[t,b]]);outward=w-center;outward/=np.linalg.norm(outward)+1e-12
   for direction in [outward,np.array([0.,0.,1.])]:
    seed=initial.copy();seed[:3]=np.clip(direction*(.07 if strict_contacts else .03),lo[:3]+1e-7,hi[:3]-1e-7);seed[6:]=np.clip(finger*(.35 if frame_mode and not active.any() else .7),lo[6:]+1e-7,hi[6:]-1e-7)
    candidate=least_squares(residual,seed,bounds=(lo,hi),max_nfev=300,diff_step=1e-4,ftol=1e-7,xtol=1e-7,gtol=1e-7,x_scale=np.r_[np.full(3,.01),np.full(3,.2),np.full(len(finger),.3)]);choices.append(candidate)
   def clearance_merit(candidate):
    update(candidate.x);_,depth=collisions();return (max(0.,depth-(.000001 if strict_contacts else .0015)),float(np.sum(residual(candidate.x)**2)))
   solution=min(choices,key=clearance_merit);update(solution.x);_,after=collisions()
  if after>(max(before,.0015) if source_sigma else before):
   solution.x=initial.copy();update(solution.x);_,after=collisions()
  newrot=Rotation.from_quat(d.qpos[adr+3:adr+7],scalar_first=True);M=np.eye(4);M[:3,:3]=Rotation.from_quat(OQ[t,b],scalar_first=True).as_matrix();M[:3,3]=O[t,b];W=np.eye(4);W[:3,:3]=newrot.as_matrix();W[:3,3]=d.qpos[adr:adr+3]
  row={'grasp_index':gi,'side':side,'object_index':b,'object':cfg['objects'][b],'held_interval':[g['start'],g['end']],'template_source_frame':t,'wrist_in_object':(np.linalg.inv(M)@W).tolist(),'finger_joints':solution.x[6:].tolist(),'maximum_penetration_m_before':before,'maximum_penetration_m_after':after,'wrist_translation_change_m':float(np.linalg.norm(solution.x[:3])),'wrist_rotation_change_rad':float(np.linalg.norm(solution.x[3:6])),'active_fingertip_errors_m':np.linalg.norm(tip_positions()-target,axis=1)[active].tolist(),'optimizer_evaluations':solution.nfev,'optimizer_success':bool(solution.success)};results.append(row);print(json.dumps({k:v for k,v in row.items() if k not in ['wrist_in_object','finger_joints']}),flush=True)
 output=os.environ.get('V2D_CONTACT_FIT_OUTPUT') or ('research/collision-transition-poses.json' if frame_mode else 'research/collision-aware-grasps.json')
 (R/output).write_text(json.dumps({'method':__doc__,'preserved_object_motion':True,'grasps':results},indent=2));raise SystemExit(0)
runner.run_training=inspect
sys.argv=[sys.argv[0],'experiment=sharpa_flash_sac',f"task.parquet='{ref.parent}'",'task.motion_speed=1.0','reset.reset_finger_openness=0.0','scene.world_count=1','logging.mode=disabled','output_dir=/episode/diagnostic-checkpoints','hydra.run.dir=/episode/hydra/collision-grasp-fit'];runpy.run_path('/workspace/scripts/train_flash_sac.py',run_name='__main__')
