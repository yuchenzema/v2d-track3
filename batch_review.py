"""Review complete physical attempts; retain earlier checkpoints when they match better."""
from pathlib import Path
import json,subprocess,sys,time,hashlib
from batch_native import containers,docker
from object_motion_quality import attach,ranking_score
from episode_clock import control_frames
BASE=Path('/home/yu_eli/v2d-track3')
def cpu(r,args,label):
 with (r/f'logs/{label}.log').open('w') as f:subprocess.run([str(r/'.venv-video/bin/python'),*args],stdout=f,stderr=subprocess.STDOUT,check=True)
def review(ep,step,gain=0.,reference_override=None,substeps_override=None,release_spec=None,checkpoint_override=None,root_override=None,container_override=None,residual_profile=None,actor_combination=None,grip_bias=None,timed_impedance=None,recording_world_count=None,tracking_feedback=None,bristle_compliance=None,wrist_calibration=None):
 if tracking_feedback or bristle_compliance or wrist_calibration:assert recording_world_count==32,'Calibrated dynamics require a jointly evaluated and recorded native cohort'
 r=Path(root_override) if root_override else BASE/'runs'/f'episode_{ep:06d}_newton_batch_20261015';cfg=json.loads((r/'episode-config.json').read_text());reference_override=reference_override or cfg.get('training_reference','track3_coherent_variant');newton=container_override or containers(r,ep)[0];cp=checkpoint_override or f'/episode/flash_chord_coherent_actor_transfer/policy_{step}.safetensors';label=f'checkpoint-{step}'+('-archived-'+hashlib.sha256(cp.encode()).hexdigest()[:10] if checkpoint_override else '')+(f'-gain-{gain}' if gain else '')+('-reference-'+reference_override if reference_override else '')+(f'-physics-{20*substeps_override}hz' if substeps_override else '')+('-release-'+hashlib.sha256(json.dumps(release_spec,sort_keys=True).encode()).hexdigest()[:8] if release_spec else '')+('-residual-'+hashlib.sha256(json.dumps(residual_profile,sort_keys=True).encode()).hexdigest()[:8] if residual_profile else '');overrides=[f"task.parquet='/episode/human_motion_data/{reference_override}/processed/sequence_id={cfg['sequence']}/robot_name=sharpa_wave'"] if reference_override else [];overrides+=([f'sim.substeps={substeps_override}'] if substeps_override else []);state=r/f'evaluation/{label}-states.npz';metric=r/f'evaluation/{label}-metrics.json';fidelity=r/f'evaluation/{label}-source-fidelity.json';single=r/f'evaluation/{label}-tracking.json'
 solver_capacity={key:int(cfg['physics_'+key]) for key in ['nconmax','njmax'] if 'physics_'+key in cfg}
 if solver_capacity:
  label+='-solver-'+hashlib.sha256(json.dumps(solver_capacity,sort_keys=True).encode()).hexdigest()[:8]
  state=r/f'evaluation/{label}-states.npz';metric=r/f'evaluation/{label}-metrics.json';fidelity=r/f'evaluation/{label}-source-fidelity.json';single=r/f'evaluation/{label}-tracking.json'
  overrides+=[f'sim.{key}={value}' for key,value in solver_capacity.items()]
 profile_env=[f'V2D_RESIDUAL_PROFILE={json.dumps(residual_profile,sort_keys=True)}'] if residual_profile else []
 if cfg.get('predecomposed_object_collisions'):overrides.append('scene.decompose_objects=false')
 if release_spec:
  import shutil
  for helper in ['release_controller.py','record_hybrid_controller.py']:shutil.copy2(BASE/helper,r/'scripts'/helper)
 if actor_combination:
  tag=hashlib.sha256(json.dumps(actor_combination,sort_keys=True).encode()).hexdigest()[:10];label+='-actors-'+tag
  state=r/f'evaluation/{label}-states.npz';metric=r/f'evaluation/{label}-metrics.json';fidelity=r/f'evaluation/{label}-source-fidelity.json';single=r/f'evaluation/{label}-tracking.json'
  profile_env+=[f'V2D_ACTOR_COMBINATION={json.dumps(actor_combination,sort_keys=True)}']
  import shutil
  for helper in ['bimanual_policy_controller.py','grip_bias_controller.py','evaluate_actor_combination.py','record_actor_combination.py','evaluate_actor_calibrated.py','record_actor_calibrated.py']:shutil.copy2(BASE/helper,r/'scripts'/helper)
 if residual_profile:
  import shutil
  for name in ['residual_gain_controller.py','evaluate_residual_profile.py','record_residual_profile.py']:shutil.copy2(BASE/name,r/'scripts'/name)
 if grip_bias:
  if residual_profile or actor_combination:raise ValueError('Evaluate grip corrections separately from gains and actor mixtures')
  import shutil
  for name in ['grip_bias_controller.py','evaluate_grip_bias.py','record_grip_bias.py']:shutil.copy2(BASE/name,r/'scripts'/name)
  label+='-grip-'+hashlib.sha256(json.dumps(grip_bias,sort_keys=True).encode()).hexdigest()[:10]
  state=r/f'evaluation/{label}-states.npz';metric=r/f'evaluation/{label}-metrics.json';fidelity=r/f'evaluation/{label}-source-fidelity.json';single=r/f'evaluation/{label}-tracking.json'
  profile_env=[f'V2D_GRIP_BIAS={json.dumps(grip_bias,sort_keys=True)}']
 if solver_capacity:
  import shutil
  shutil.copy2(BASE/'evaluate_reference_controller.py',r/'scripts/evaluate_reference_controller.py')
  profile_env+=[f'V2D_SIM_{key.upper()}={value}' for key,value in solver_capacity.items()]
 if cfg.get('predecomposed_object_collisions'):
  profile_env.append('V2D_SCENE_DECOMPOSE_OBJECTS=false')
 if timed_impedance:
  if gain or release_spec or residual_profile or grip_bias:raise ValueError('Combine timed impedance only with an explicitly declared learned-actor combination')
  import shutil
  for name in ['timed_wrist_impedance_controller.py','evaluate_timed_impedance.py','record_timed_impedance.py','evaluate_actor_timed_impedance.py','record_actor_timed_impedance.py']:shutil.copy2(BASE/name,r/'scripts'/name)
  label+='-timed-wrist-'+hashlib.sha256(json.dumps(timed_impedance,sort_keys=True).encode()).hexdigest()[:10]
  state=r/f'evaluation/{label}-states.npz';metric=r/f'evaluation/{label}-metrics.json';fidelity=r/f'evaluation/{label}-source-fidelity.json';single=r/f'evaluation/{label}-tracking.json'
  profile_env+=[f'V2D_TIMED_WRIST_IMPEDANCE_PROFILE={json.dumps(timed_impedance,sort_keys=True)}']
 if recording_world_count:
  assert recording_world_count==32,'Keep the existing32-world review protocol and predetermined world0'
  import shutil
  shutil.copy2(BASE/'record_native_cohort.py',r/'scripts/record_native_cohort.py')
  label+='-cohort32-world0';state=r/f'evaluation/{label}-states.npz';metric=r/f'evaluation/{label}-metrics.json';fidelity=r/f'evaluation/{label}-source-fidelity.json';single=r/f'evaluation/{label}-tracking.json'
  if tracking_feedback:
   assert not any([actor_combination,residual_profile,grip_bias,timed_impedance]),'Validate tracking feedback separately from other actor action calibrations'
   for name in ['robot_object_tracking_feedback.py','evaluate_reference_controller.py']:shutil.copy2(BASE/name,r/'scripts'/name)
   profile_env.append('V2D_ROBOT_TRACKING_FEEDBACK='+json.dumps(tracking_feedback,sort_keys=True))
   label+='-robot-tracking-'+hashlib.sha256(json.dumps(tracking_feedback,sort_keys=True).encode()).hexdigest()[:10]
   state=r/f'evaluation/{label}-states.npz';metric=r/f'evaluation/{label}-metrics.json';fidelity=r/f'evaluation/{label}-source-fidelity.json';single=r/f'evaluation/{label}-tracking.json'
  for spec,key,helper in [(bristle_compliance,'V2D_BRISTLE_COMPLIANCE','brush_bristle_compliance.py'),(wrist_calibration,'V2D_ROBOT_WRIST_CALIBRATION','robot_wrist_calibration.py')]:
   if spec:
    shutil.copy2(BASE/helper,r/'scripts'/helper);shutil.copy2(BASE/'evaluate_reference_controller.py',r/'scripts/evaluate_reference_controller.py')
    tag='bristles' if key=='V2D_BRISTLE_COMPLIANCE' else 'motors'
    profile_env.append(key+'='+json.dumps(spec,sort_keys=True));label+='-'+tag+'-'+hashlib.sha256(json.dumps(spec,sort_keys=True).encode()).hexdigest()[:8]
    state=r/f'evaluation/{label}-states.npz';metric=r/f'evaluation/{label}-metrics.json';fidelity=r/f'evaluation/{label}-source-fidelity.json';single=r/f'evaluation/{label}-tracking.json'
  if not state.exists() or not metric.exists():
   docker(['exec',newton,'env',*profile_env,f'V2D_WRIST_VELOCITY_GAIN={gain}',*([f'V2D_RELEASE_SPEC={json.dumps(release_spec)}'] if release_spec else []),*([f'V2D_SIM_SUBSTEPS={substeps_override}'] if substeps_override else []),*([f"V2D_REFERENCE_OVERRIDE=/episode/human_motion_data/{reference_override}/processed/sequence_id={cfg['sequence']}/robot_name=sharpa_wave"] if reference_override else []),f'V2D_RECORD_COHORT_SIZE={recording_world_count}',f'V2D_STATE_OUTPUT=/episode/evaluation/{state.name}','python','-u','/episode/scripts/record_native_cohort.py',f'evaluation.checkpoint={cp}','evaluation.world_count=32','logging.mode=disabled',f'evaluation.metrics_output=/episode/evaluation/{metric.name}',f'hydra.run.dir=/episode/hydra/{label}-record'],r/f'logs/{label}-record.log')
 if not metric.exists():docker(['exec',newton,'env',*profile_env,f'V2D_WRIST_VELOCITY_GAIN={gain}',*([f'V2D_RELEASE_SPEC={json.dumps(release_spec)}'] if release_spec else []),*([f'V2D_SIM_SUBSTEPS={substeps_override}'] if substeps_override else []),*([f"V2D_REFERENCE_OVERRIDE=/episode/human_motion_data/{reference_override}/processed/sequence_id={cfg['sequence']}/robot_name=sharpa_wave"] if reference_override else []),'python','-u','/episode/scripts/evaluate_actor_calibrated.py' if actor_combination and residual_profile else '/episode/scripts/evaluate_actor_timed_impedance.py' if timed_impedance and actor_combination else '/episode/scripts/evaluate_timed_impedance.py' if timed_impedance else '/episode/scripts/evaluate_grip_bias.py' if grip_bias else '/episode/scripts/evaluate_actor_combination.py' if actor_combination else '/episode/scripts/evaluate_residual_profile.py' if residual_profile else '/episode/scripts/evaluate_reference_controller.py',f'evaluation.checkpoint={cp}','evaluation.world_count=32','logging.mode=disabled',f'evaluation.metrics_output=/episode/evaluation/{metric.name}',f'hydra.run.dir=/episode/hydra/{label}-eval'],r/f'logs/{label}-eval.log')
 if not state.exists():docker(['exec',newton,'env',*profile_env,f'V2D_WRIST_VELOCITY_GAIN={gain}',*([f'V2D_RELEASE_SPEC={json.dumps(release_spec)}'] if release_spec else []),f'V2D_STATE_OUTPUT=/episode/evaluation/{state.name}','python','-u','/episode/scripts/record_actor_calibrated.py' if actor_combination and residual_profile else '/episode/scripts/record_actor_timed_impedance.py' if timed_impedance and actor_combination else '/episode/scripts/record_timed_impedance.py' if timed_impedance else '/episode/scripts/record_grip_bias.py' if grip_bias else '/episode/scripts/record_actor_combination.py' if actor_combination else '/episode/scripts/record_residual_profile.py' if residual_profile else '/episode/scripts/record_hybrid_controller.py' if release_spec else '/episode/scripts/record_velocity_controller.py' if gain else '/episode/scripts/record_policy_states.py',f'evaluation.checkpoint={cp}',*overrides,'viewer.backend=none',f'hydra.run.dir=/episode/hydra/{label}-record'],r/f'logs/{label}-record.log')
 native_report=json.loads(metric.read_text())
 expected=Path('/episode')/f"human_motion_data/{reference_override}/processed/sequence_id={cfg['sequence']}/robot_name=sharpa_wave/motion.parquet"
 actual=Path(native_report['parquet']);actual=actual if actual.suffix=='.parquet' else actual/'motion.parquet'
 assert actual==expected,('Native evaluation used another motion reference',native_report['parquet'],str(expected))
 import numpy as np
 data=np.load(state)
 if not np.isfinite(data['body_poses']).all():
  metrics=json.loads(metric.read_text())['metrics']
  return {'checkpoint':cp,'environment_steps':step,'additional_environment_steps':step-cfg.get('training_initial_counter',250003456 if len(cfg['objects'])==2 else 0),'completion_rate_32_worlds':metrics['chord_sr'],'source_agreement_score':-1.,'source_mean_object_iou':0.,'source_worst_phase_object_iou':0.,'recorded_states':str(state.relative_to(r)),'single_attempt_tracking':{'passed_all_frames':False,'all_frames_finite':False},'wrist_reference_velocity_gain':gain,'rejected':'Non-finite physical state; candidate cannot be selected while finite candidates exist.'}
 if not fidelity.exists():cpu(r,[str(r/'scripts/assess_source_fidelity.py'),'--states',str(state),'--output',f'{label}-source-fidelity'],f'{label}-source')
 if not single.exists():cpu(r,[str(r/'scripts/assess_single_attempt.py'),'--states',str(state),'--reference',reference_override or cfg.get('training_reference','track3_coherent_variant'),'--output',str(single)],f'{label}-tracking')
 metrics=json.loads(metric.read_text())['metrics'];source=json.loads(fidelity.read_text());tracking=json.loads(single.read_text());agg=source['aggregate'];mean=sum(v['iou'] for v in agg['objects'].values())/len(cfg['objects']);worst=min(v['mean_iou'] for phase in source['phases'] for v in phase['objects'].values());score=mean+.25*worst
 import numpy as np
 data=np.load(state);assert np.isfinite(data['body_poses']).all() and np.all(data['object_assistance']==0) and np.array_equal(data['reference_frames'],np.arange(control_frames(cfg)))
 row={'checkpoint':cp,'checkpoint_sha256':hashlib.sha256((r/cp.removeprefix('/episode/')).read_bytes()).hexdigest(),'environment_steps':step,'additional_environment_steps':step-cfg.get('training_initial_counter',250003456 if len(cfg['objects'])==2 else 0),'completion_rate_32_worlds':metrics['chord_sr'],'reference_object_position_error_m':metrics['object_position_error_m'],'source_agreement_score':score,'source_mean_object_iou':mean,'source_worst_phase_object_iou':worst,'source_metrics':agg,'recorded_states':str(state.relative_to(r)),'single_attempt_tracking':tracking,'wrist_reference_velocity_gain':gain,'inference_reference':reference_override or cfg.get('training_reference','track3_coherent_variant'),'physics_substeps':substeps_override or cfg.get('physics_substeps',5),'robot_release_controller_spec':release_spec,'residual_gain_profile':residual_profile,'actor_combination':actor_combination}
 if solver_capacity:row['solver_capacity']=solver_capacity
 if cfg.get('predecomposed_object_collisions'):row['scene_parameters']={'decompose_objects':False}
 if grip_bias:row['grip_bias_profile']=grip_bias
 if timed_impedance:row['timed_wrist_impedance_profile']=timed_impedance
 if recording_world_count:row['recording_protocol']={'simulation_world_count':32,'recorded_world_index':0,'world_selection':'Predetermined0; no success-based selection','metrics_and_video_share_the_same_physical_cohort':True}
 if tracking_feedback:
  assert recording_world_count==32,'Feedback requires a jointly evaluated and recorded native cohort'
  row['robot_tracking_feedback_profile']=tracking_feedback
 if bristle_compliance:row['brush_bristle_compliance_profile']=bristle_compliance
 if wrist_calibration:row['robot_wrist_calibration_profile']=wrist_calibration
 if cfg.get('requires_no_penetration_for_publication'):
  from rollout_collision_validation import audit_rollout
  row['collision_validation']=audit_rollout(r,state,label,reference_override)
 return attach(r,row)
def choose(rows):
 rows=[r for r in rows if r['single_attempt_tracking'].get('all_frames_finite',True)]
 if not rows:raise RuntimeError('No finite physical attempt exists; reference or training requires repair')
 from rollout_collision_validation import collision_verified
 collision_safe=[r for r in rows if collision_verified(r.get('collision_validation') or {},(r.get('collision_validation') or {}).get('states_sha256'))]
 if collision_safe:rows=collision_safe
 valid=[r for r in rows if r['single_attempt_tracking']['passed_all_frames'] and r['completion_rate_32_worlds']>=.25]
 rank=ranking_score
 if valid:return max(valid,key=rank)
 best_rate=max(r['completion_rate_32_worlds'] for r in rows);return max([r for r in rows if r['completion_rate_32_worlds']>=best_rate-.125],key=rank)
def monitor(ep,root_override=None,container_override=None,stop_event=None):
 from functools import partial
 inspect=partial(review,ep,root_override=root_override,container_override=container_override)
 r=Path(root_override) if root_override else BASE/'runs'/f'episode_{ep:06d}_newton_batch_20261015';folder=r/'flash_chord_coherent_actor_transfer';rp=r/'checkpoint-reviews.json';rows=json.loads(rp.read_text()) if rp.exists() else []
 while True:
  if stop_event is not None and stop_event.is_set():raise RuntimeError(f'Episode {ep} learner exited before checkpoint review completed')
  known={x['environment_steps'] for x in rows};checkpoints=sorted(folder.glob('policy_*.safetensors'),key=lambda p:int(p.stem.split('_')[-1]))
  for cp in checkpoints:
   step=int(cp.stem.split('_')[-1])
   if step in known or not cp.with_name('state_'+str(step)+'.safetensors').exists() or time.time()-cp.stat().st_mtime<5:continue
   row=inspect(step);rows.append(row);rp.write_text(json.dumps(rows,indent=2));print(ep,json.dumps({k:row[k] for k in ['additional_environment_steps','completion_rate_32_worlds','source_mean_object_iou','source_worst_phase_object_iou']}),flush=True)
   # Keep best and latest, plus the immediately preceding checkpoint. All old
   # episode 1/5 artifacts remain untouched. Every removed candidate is reviewed.
   best=choose(rows);keep={best['environment_steps'],*sorted(x['environment_steps'] for x in rows)[-2:]}
   # Preserve every potentially selectable score/completion tradeoff: a later
   # high completion rate may change the eligibility threshold.
   for candidate in rows:
    group=bool(candidate['single_attempt_tracking'].get('passed_all_frames')) and candidate['completion_rate_32_worlds']>=.25
    dominated=any((bool(other['single_attempt_tracking'].get('passed_all_frames')) and other['completion_rate_32_worlds']>=.25)==group and other['completion_rate_32_worlds']>=candidate['completion_rate_32_worlds'] and other['source_agreement_score']>=candidate['source_agreement_score'] and ranking_score(other)>=ranking_score(candidate) and (other['source_agreement_score']>candidate['source_agreement_score'] or other['completion_rate_32_worlds']>candidate['completion_rate_32_worlds']) for other in rows)
    if not dominated:keep.add(candidate['environment_steps'])
   for candidate in rows:
    if candidate['environment_steps'] in keep:continue
    for kind in ['state','policy']:
     old=folder/f"{kind}_{candidate['environment_steps']}.safetensors"
     if old.exists():old.unlink()
  complete=(folder/'TRAINING_COMPLETED').exists()
  if complete and all(int(p.stem.split('_')[-1]) in {x['environment_steps'] for x in rows} for p in folder.glob('policy_*.safetensors')):break
  log=r/'logs/newton-training.log'
  if log.exists() and 'Traceback' in log.read_text()[-5000:]:raise RuntimeError(f'Episode {ep} training failed')
  if stop_event is not None:stop_event.wait(15)
  else:time.sleep(15)
 assert rows
 best=choose(rows);controller=[best]
 cfg=json.loads((r/'episode-config.json').read_text())
 for gain in ([] if cfg.get('validated_video_pipeline') else [.25,.5]):
  candidate=inspect(best['environment_steps'],gain);controller.append(candidate)
 if ep==3 and (r/'research/collision-floor-projection.json').exists():
  for gain in [0.,.5]:controller.append(inspect(best['environment_steps'],gain,'track3_collision_variant'))
 eligible=[best]+[x for x in controller[1:] if x['single_attempt_tracking']['passed_all_frames'] and x['completion_rate_32_worlds']>=max(.25,best['completion_rate_32_worlds']-.125)]
 from rollout_collision_validation import collision_verified
 collision_safe=[x for x in eligible if collision_verified(x.get('collision_validation') or {},(x.get('collision_validation') or {}).get('states_sha256'))]
 if collision_safe:eligible=collision_safe
 selected=max(eligible,key=ranking_score);(r/'final-controller-selection.json').write_text(json.dumps({'criterion':'Choose entire unassisted attempts; prefer passing all native tracking thresholds, then source object overlap over every frame and the worst phase. Feedforward accepted only without material completion regression. No time warping, resets, object guidance or frame splicing.','selected':selected,'checkpoint_candidates':rows,'controller_candidates':controller},indent=2));print(ep,'selected',selected['checkpoint'],'gain',selected['wrist_reference_velocity_gain'],flush=True)
if __name__=='__main__':
 if len(sys.argv)>2:print(json.dumps(review(int(sys.argv[1]),int(sys.argv[2])),indent=2))
 else:monitor(int(sys.argv[1]))
