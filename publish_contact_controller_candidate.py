"""Publish a full, independently checked controller candidate only after improvement."""
import argparse,copy,json,shutil
from pathlib import Path
from improvement_common import BASE,BATCH,baseline,better,digest,save,utc,verify_preserved
from batch_native import docker
from render_improved_controller import render

def publish(ep,review_file,label,root_override=None):
 assert ep in [10,16,30,32] and label.replace('-','').isalnum()
 r=Path(root_override) if root_override else BASE/f'runs/episode_{ep:06d}_newton_grasp_curriculum_20261020';cfg=json.loads((r/'episode-config.json').read_text());assert cfg['episode']==ep
 selected=copy.deepcopy(json.loads(Path(review_file).read_text()));previous=baseline(ep)
 assert not selected.get('reference_protocol_mismatch'),'Rerun mismatched reference diagnostics before publication'
 assert better(selected,previous),'No measured per-object improvement: candidate remains diagnostic'
 matched_baseline=None
 if selected.get('recording_protocol'):
  from batch_review import review
  matched_baseline=review(ep,previous['environment_steps'],gain=previous.get('wrist_reference_velocity_gain',0.),
   reference_override=previous['inference_reference'],substeps_override=previous['physics_substeps'],
   checkpoint_override=previous['checkpoint'],root_override=cfg['baseline_run'],recording_world_count=32)
  assert better(selected,matched_baseline),'No improvement under the same native cohort recording protocol'
 dst=r/'research'/f'controller-release-{label}';dst.mkdir(exist_ok=True)
 primary=r/selected['checkpoint'].removeprefix('/episode/');target=dst/primary.name
 if not target.exists():shutil.copy2(primary,target)
 assert digest(target)==selected['checkpoint_sha256'];selected['checkpoint']='/episode/'+str(target.relative_to(r))
 if selected.get('actor_combination'):
  spec=selected['actor_combination'];alternate=r/spec['alternate_checkpoint'].removeprefix('/episode/');alt=dst/('alternate-'+alternate.name)
  if not alt.exists():shutil.copy2(alternate,alt)
  assert digest(alt)==spec['alternate_checkpoint_sha256'];spec['alternate_checkpoint']='/episode/'+str(alt.relative_to(r))
 states=r/selected['recorded_states'];snapshot=dst/'policy-states.npz'
 if not snapshot.exists():shutil.copy2(states,snapshot)
 assert digest(states)==digest(snapshot);selected['recorded_states']=str(snapshot.relative_to(r))
 # A whole-source32-world review is followed by a larger, unchanged native cohort.
 env=[f'V2D_WRIST_VELOCITY_GAIN={selected.get("wrist_reference_velocity_gain",0.)}',f'V2D_SIM_SUBSTEPS={selected["physics_substeps"]}',f'V2D_REFERENCE_OVERRIDE=/episode/human_motion_data/{selected["inference_reference"]}/processed/sequence_id={cfg["sequence"]}/robot_name=sharpa_wave',
  *[f'V2D_SIM_{key.upper()}={value}' for key,value in selected.get('solver_capacity',{}).items()]]
 script='evaluate_reference_controller.py'
 for key,var in [('robot_wrist_calibration_profile','V2D_ROBOT_WRIST_CALIBRATION'),('brush_bristle_compliance_profile','V2D_BRISTLE_COMPLIANCE')]:
  if selected.get(key):env.append(var+'='+json.dumps(selected[key],sort_keys=True))
 if selected.get('scene_parameters'):env.append('V2D_SCENE_DECOMPOSE_OBJECTS='+str(selected['scene_parameters']['decompose_objects']).lower())
 for key,var,wrapper in [('actor_combination','V2D_ACTOR_COMBINATION','evaluate_actor_combination.py'),('timed_wrist_impedance_profile','V2D_TIMED_WRIST_IMPEDANCE_PROFILE','evaluate_timed_impedance.py'),('residual_gain_profile','V2D_RESIDUAL_PROFILE','evaluate_residual_profile.py'),('grip_bias_profile','V2D_GRIP_BIAS','evaluate_grip_bias.py'),('robot_tracking_feedback_profile','V2D_ROBOT_TRACKING_FEEDBACK','evaluate_reference_controller.py')]:
  if selected.get(key):env.append(var+'='+json.dumps(selected[key],sort_keys=True));script=wrapper
 if selected.get('actor_combination') and selected.get('timed_wrist_impedance_profile'):script='evaluate_actor_timed_impedance.py'
 if selected.get('actor_combination') and selected.get('residual_gain_profile'):script='evaluate_actor_calibrated.py'
 if selected.get('robot_release_controller_spec'):env.append('V2D_RELEASE_SPEC='+json.dumps(selected['robot_release_controller_spec']))
 metrics=dst/'native-128-metrics.json'
 if not metrics.exists():
  docker(['exec',cfg['native_container'],'env',*env,'python','-u','/episode/scripts/'+script,
   'evaluation.checkpoint='+selected['checkpoint'],'evaluation.world_count=128','logging.mode=disabled',
   'evaluation.metrics_output=/episode/'+str(metrics.relative_to(r)),f'hydra.run.dir=/episode/hydra/controller-release-{label}-128'],dst/'native-128.log')
 cohort=json.loads(metrics.read_text());assert cohort['step_count']==2*(cfg['frames']-1) and cohort['start_frame']==0
 expected=Path('/episode')/f"human_motion_data/{selected['inference_reference']}/processed/sequence_id={cfg['sequence']}/robot_name=sharpa_wave/motion.parquet"
 actual=Path(cohort['parquet']);actual=actual if actual.suffix=='.parquet' else actual/'motion.parquet'
 assert actual==expected,'Independent validation used another motion reference'
 baseline128=json.loads((Path(cfg['baseline_run'])/'evaluation/final-metrics.json').read_text())['metrics']['chord_sr']
 from video_catalog import catalog
 accepted=catalog(ep,cfg['baseline_run'],True).get('best')
 if accepted:baseline128=max(baseline128,accepted['quality']['completion_rate_against_video_estimate'])
 assert cohort['metrics']['chord_sr']>=baseline128-.0625,'Larger validation cohort regressed'
 rendered=render(ep,selected,dst,root_override=r,container_override=cfg['native_container']);verify_preserved(ep)
 training_complete=(r/'flash_chord_coherent_actor_transfer/TRAINING_COMPLETED').exists()
 quality={'all_internal_frame_checks_passed':selected['object_motion_quality']['every_object_passes_all_local_frames'] and cohort['metrics']['chord_sr']==1.,
  'completion_rate_against_video_estimate':cohort['metrics']['chord_sr'],'full_human_video_match_verified':False,'training_continues':not training_complete}
 result={'episode':ep,'status':'completed','release_type':'controller','updated_utc':utc(),'evaluation_run':str(r),
  'experiment':'Source-video contact optimization; complete unassisted learned-policy rollout',
  'selected':selected,'baseline':previous,'measured_improvement':True,'rendered':rendered,
  'native_128_world_evaluation':cohort,'quality_assessment':quality,'training_completed':training_complete,'full_human_motion_match_verified':False}
 if matched_baseline:result['matched_recording_protocol_baseline']=matched_baseline
 output=BATCH/'improvements_20261020_contacts'/f'episode_{ep:06d}-controller-{label}-result.json';save(output,result)
 print('Verified full-source controller videos published:',output,flush=True)
 return result

if __name__=='__main__':
 p=argparse.ArgumentParser(description=__doc__);p.add_argument('episode',type=int);p.add_argument('review',type=Path);p.add_argument('label');a=p.parse_args();publish(a.episode,a.review,a.label)
