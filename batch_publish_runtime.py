"""Publish executable inference settings without changing the original training checkpoint metadata."""
from pathlib import Path
import json,hashlib,time,sys,shutil
BASE=Path('/home/yu_eli/v2d-track3')
def publish(r):
 reportfile=r/'result-report.json';manifestfile=r/'deliverables/manifest.json'
 if not reportfile.exists() or not manifestfile.exists() or time.time()-reportfile.stat().st_mtime<10:return
 c=json.loads((r/'episode-config.json').read_text());selection=json.loads((r/'final-controller-selection.json').read_text())['selected'];training=json.loads((r/'coherent-training-config.json').read_text());reference=selection.get('inference_reference',c.get('training_reference','track3_coherent_variant'));substeps=selection.get('physics_substeps',c.get('physics_substeps',5));key=[selection['checkpoint'],reference,substeps,selection['wrist_reference_velocity_gain']];report=json.loads(reportfile.read_text())
 key=[7,*key,selection.get('robot_release_controller_spec')]
 if report.get('runtime_configuration',{}).get('selection_key')==key:return
 previous=r/'research/result-report-before-runtime-v2.json'
 if not previous.exists():shutil.copy2(reportfile,previous)
 state=r/selection['recorded_states'];data={'selection_key':key,'checkpoint':selection['checkpoint'],'checkpoint_metadata_preserved':True,'saved_training':{'physics_substeps':training['env']['sim']['substeps'],'physics_integration_hz':training['env']['sim']['substeps']*20,'control_hz':20},'inference':{'reference':reference,'physics_substeps':substeps,'physics_integration_hz':substeps*20,'control_hz':20,'motion_speed':(1. if c.get('validated_video_pipeline') else .5),'wrist_reference_velocity_gain':selection['wrist_reference_velocity_gain'],'object_assistance':0.,'failure_resets':False},'selected_recorded_states':selection['recorded_states'],'selected_states_sha256':hashlib.sha256(state.read_bytes()).hexdigest(),'reproduce_command':['python3',str(r/'scripts/reproduce_selected_policy.py')],'reproducibility_note':'Reproduction executes the learned policy with the declared integration settings. Contact simulation can vary; published views replay one identical, fully recorded physical attempt.'};(r/'runtime-config.json').write_text(json.dumps(data,indent=2));shutil.copy2(BASE/'reproduce_selected_policy.py',r/'scripts/reproduce_selected_policy.py');manifest=json.loads(manifestfile.read_text())
 data['inference']['robot_release_controller_spec']=selection.get('robot_release_controller_spec');reference_path=next((r/f'human_motion_data/{reference}/processed').rglob('motion.parquet'));data['inference']['reference_parquet_sha256']=hashlib.sha256(reference_path.read_bytes()).hexdigest();data['checkpoint_sha256']=hashlib.sha256((r/selection['checkpoint'].removeprefix('/episode/')).read_bytes()).hexdigest();(r/'runtime-config.json').write_text(json.dumps(data,indent=2))
 if selection.get('robot_release_controller_spec'):
  for helper in ['release_controller.py','record_hybrid_controller.py']:shutil.copy2(BASE/helper,r/'scripts'/helper)
 for name,target in [('runtime-config.json',r/'runtime-config.json'),('reproduce_policy.py',r/'scripts/reproduce_selected_policy.py'),('trained_policy_source_camera.mp4',r/'videos/trained_policy_source_camera.mp4'),('reconstruction_reference.mp4',r/'videos/reconstruction_reference.mp4')]:
  link=r/'deliverables'/name
  if link.is_symlink():link.unlink()
  link.symlink_to(Path('..')/target.relative_to(r));manifest[name]={'path':str(target.relative_to(r)),'bytes':target.stat().st_size,'sha256':hashlib.sha256(target.read_bytes()).hexdigest()}
 report['runtime_configuration']=data
 metrics=r/'flash_chord_coherent_actor_transfer/metrics.jsonl'
 if metrics.exists():report['training']['actual_additional_environment_steps']=json.loads(metrics.read_text().splitlines()[-1])['additional_environment_steps']
 report['training']['initialization']=report['training'].get('initialization') or ('fresh single-object learner' if len(c['objects'])==1 else 'preferred actor transfer; critic, target critic, temperature and optimizer initialized fresh')
 public=[name for name in c['objects'] if name!='wooden_spoon'];report['training']['reference_source']=f"All {c['frames']} original evaluation frames, calibrated camera/stereo, released public assets {public}, and an episode-specific video-reconstructed spoon when present; contact and geometry refinements documented in this run."
 quality=report['quality_assessment'];quality['all_internal_frame_checks_passed']=quality.get('all_internal_frame_checks_passed',quality['full_human_video_match_verified']);quality['full_human_video_match_verified']=False;quality['match_interpretation']='Internal thresholds and inferred poses do not establish exact human-video or robot-hand pixel equivalence.'
 archive=r/'research/checkpoint-archive/manifest.json'
 if archive.exists():report['earlier_checkpoint_archive']=str(archive.relative_to(r))
 for key,filename in [('physical_training_reference_source_fidelity','evaluation/physical-reference-source-fidelity.json'),('observed_rest_stability','reconstruction/observed-rest-stability.json'),('table_gauge_refit','evaluation/table-gauge-refit.json'),('training_resume_sessions','research/training-resume-sessions.json'),('image_geometry_reconstruction','research/image-geometry-selection.json'),('release_contact_targets','research/release-contact-targets.json'),('validated_early_stopping','research/training-early-stop.json')]:
  p=r/filename
  if p.exists():report[key]=json.loads(p.read_text())
 for key,filename,excluded in [('physical_object_pose_refinement','research/native-object-6d-refinement.json','frames'),('precise_rack_collision','research/precise-rack-collision.json','parts')]:
  p=r/filename
  if p.exists():report[key]={k:v for k,v in json.loads(p.read_text()).items() if k!=excluded}
 for key,filename in [('source_support_geometry_refinement','research/support-geometry-selection.json'),('initial_support_reference_refinement',f"research/{c.get('training_reference','track3_coherent_variant')}-initial-support-selection.json"),('passive_rest_integration_selection','research/passive-rest-integration-selection.json'),('early_hand_track_repair','research/early-left-hand-repair.json'),('early_hand_reference_splice','research/early-hand-reference-splice.json'),('gpu_runtime_recovery','research/gpu-container-recovery.json')]:
  path=r/filename
  if path.exists():report[key]=json.loads(path.read_text())
 releases=[json.loads(p.read_text()) for p in sorted((r/'research').glob('continuous-release-selection-v*.json')) if '-inference' not in p.name]
 if releases:report['continuous_release_reference_reviews']=releases
 for key,filename in [('terminal_support_diagnostic','evaluation/terminal-support-diagnostic.json'),('terminal_support_reference_refinement','research/terminal-support-reference-selection.json'),('source_aligned_grasp_refinement','research/source-aligned-grasp-selection.json'),('integration_controller_optimization','research/integration-controller-selection.json'),('saucepan_handle_reconstruction','research/saucepan-handle-selection.json')]:
  path=r/filename
  if path.exists():report[key]=json.loads(path.read_text())
 for key,filename in [('occluded_wearer_hand_repair','research/occluded-wearer-hand-repair.json'),('precise_supported_rack_trial','research/precise-supported-rack-selection.json'),('earlier_supported_placement_retained','research/terminal-support-later-search-review.json'),('supported_blue_cup_refinement','research/terminal-support-blue-cup-selection.json'),('temporal_reference_refinement','research/temporal-reference-selection.json'),('reference_training_session_transition','research/reference-training-session-transition.json'),('earlier_training_session_comparison','research/prior-training-session-comparison.json')]:
  path=r/filename
  if path.exists():report[key]=json.loads(path.read_text())
 if report.get('occluded_wearer_hand_repair',{}).get('source_hidden_motion_is_inferred'):
  limitation='The wearer hand is absent from the video during the documented occlusion interval; its hidden free motion is inferred, and background-person detections are excluded.'
  if limitation not in report['limitations']:report['limitations'].append(limitation)
 temp=reportfile.with_suffix('.runtime.tmp');temp.write_text(json.dumps(report,indent=2));temp.replace(reportfile);manifest['report.json']['bytes']=reportfile.stat().st_size;manifest['report.json']['sha256']=hashlib.sha256(reportfile.read_bytes()).hexdigest();tmp=manifestfile.with_suffix('.tmp');tmp.write_text(json.dumps(manifest,indent=2));tmp.replace(manifestfile);print(c['episode'],'runtime settings published',data['inference'],flush=True)
if __name__=='__main__':
 while True:
  for r in sorted((BASE/'runs').glob('episode_*_newton_batch_20261015')):publish(r)
  if '--watch' not in sys.argv:break
  time.sleep(30)
