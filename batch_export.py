"""Export synchronized source/physics comparisons and auditable per-episode deliverables."""
from pathlib import Path
import json,sys,hashlib,subprocess,shutil
import os
import batch_worker as w
from batch_native import docker,containers
from episode_clock import control_frames,motion_speed
from legacy_source_frame_playback import prepare_script
BASE=w.BASE

def archive_file(source,target):
 if target.exists():return
 try:os.link(source.resolve(),target)
 except OSError:shutil.copy2(source.resolve(),target)

def export(ep,root_override=None,container_override=None):
 if root_override:
  r=Path(root_override);c=json.loads((r/'episode-config.json').read_text())
 else:
  from batch_compare_prior_training_session import compare
  compare(ep)
  from batch_inference_optimization import optimize
  optimize(ep)
  r,c=w.context(ep)
 from export_reference_diagnostics import ensure as ensure_diagnostics
 ensure_diagnostics(r)
 if c.get('official_evaluation_required'):
  from official_policy_suite import select_recorded_attempt
  select_recorded_attempt(r)
 newton=container_override or containers(r,ep)[0];T=c['frames'];N=control_frames(c);speed=motion_speed(c);selection=json.loads((r/'final-controller-selection.json').read_text());chosen=selection['selected'];checkpoint=chosen['checkpoint'];gain=chosen['wrist_reference_velocity_gain'];state=r/chosen['recorded_states'];state_rel='/episode/'+str(state.relative_to(r));video=r/'videos';evaluation=r/'evaluation'
 physics_substeps=chosen.get('physics_substeps',c.get('physics_substeps',5))
 if c.get('validated_video_pipeline'):
  from improvement_common import digest,save
  identity={'checkpoint':digest(r/checkpoint.removeprefix('/episode/')),'states':digest(state),'control_frames':N,'physics_substeps':physics_substeps}
  marker=r/'research/export-source-identity.json'
  if marker.exists() and json.loads(marker.read_text())!=identity:
   previous=json.loads(marker.read_text());archive=r/'research/export-history'/previous['states'][:16]
   paths=[*video.glob('*'),evaluation/'final-metrics.json',evaluation/'final-trajectories.parquet',evaluation/'source_camera_policy_states.npz',evaluation/'reconstruction-rendered-states.npz',r/'result-report.json',r/'runtime-config.json',*list((r/'deliverables').glob('*'))]
   for path in paths:
    if path.is_file():
     target=archive/path.relative_to(r);target.parent.mkdir(parents=True,exist_ok=True)
     archive_file(path,target)
     path.unlink()
  save(marker,identity)
 ref=chosen.get('inference_reference',c.get('training_reference','track3_coherent_variant'));inference_override=[f"task.parquet='/episode/human_motion_data/{ref}/processed/sequence_id={c['sequence']}/robot_name=sharpa_wave'"]
 def native(args,label):docker(['exec',newton,*args],r/f'logs/{label}.log')
 import numpy as np,cv2
 record=np.load(state);assert len(record['body_poses'])==N and np.isfinite(record['body_poses']).all() and np.all(record['object_assistance']==0)
 checksums=json.loads((r/'coherent-input-checksums.json').read_text())
 for path,digest in checksums.items():assert hashlib.sha256((r/path).read_bytes()).hexdigest()==digest,path
 if not (evaluation/'final-metrics.json').exists():native(['env',f'V2D_WRIST_VELOCITY_GAIN={gain}',f'V2D_SIM_SUBSTEPS={physics_substeps}',f"V2D_REFERENCE_OVERRIDE=/episode/human_motion_data/{ref}/processed/sequence_id={c['sequence']}/robot_name=sharpa_wave",'python','-u','/episode/scripts/evaluate_reference_controller.py',f'evaluation.checkpoint={checkpoint}','evaluation.world_count=128','logging.mode=disabled','evaluation.metrics_output=/episode/evaluation/final-metrics.json','evaluation.object_trajectories_output=/episode/evaluation/final-trajectories.parquet','hydra.run.dir=/episode/hydra/final-evaluation'],'final-evaluation')
 shared=['viewer.backend=mp4','viewer.width=1024','viewer.height=768',f'viewer.num_frames={N}','viewer.video_fps=20','viewer.realtime=false','markers.enabled=false']
 for script,name,extra in [('render_source_camera.py','trained_policy_source_camera',[]),('render_recorded_oblique.py','trained_policy_rollout',['viewer.width=1280','viewer.height=720'])]:
  if not (video/f'{name}.mp4').exists():native(['env',f'V2D_REPLAY_STATE_FILE={state_rel}','python','-u',f'/episode/scripts/{script}',f'evaluation.checkpoint={checkpoint}',*shared,*extra,f'viewer.output_path=/episode/videos/{name}.mp4',f'hydra.run.dir=/episode/hydra/{name}'],name)
  protocol=video/f'{name}.protocol.json';d=json.loads(protocol.read_text());d.update(render_mode='Exact replay of the selected genuine unassisted Newton physical rollout',state_sha256=hashlib.sha256(state.read_bytes()).hexdigest(),selected_policy_sha256=chosen.get('checkpoint_sha256'),object_assistance=0.,failure_resets=False,motion_speed=speed,source_frame_count=T,control_frame_count=N,wrist_velocity_gain=gain,physics_integration_hz=20*physics_substeps,physics_substeps=physics_substeps);tmp=protocol.with_suffix('.host.tmp');tmp.write_text(json.dumps(d,indent=2));tmp.replace(protocol)
 rendered=np.load(evaluation/'source_camera_policy_states.npz');assert np.array_equal(rendered['body_poses'],record['body_poses'])
 old=BASE/'runs/episode_000005_newton_20261015/scripts/record_reconstruction.py';s=old.read_text().replace('episode_000005',c['sequence']);
 if c.get('validated_video_pipeline'):s=s.replace("'task.motion_speed=0.5'","'task.motion_speed=1.0','task.source_frame_playback=true'")
 s=prepare_script(s,c)
 if c.get('validated_video_pipeline'):shutil.copy2(BASE/'legacy_source_frame_playback.py',r/'scripts/legacy_source_frame_playback.py')
 (r/'scripts/record_reconstruction.py').write_text(s)
 if not (evaluation/'reconstruction-reference-states.npz').exists():native(['python','-u','/episode/scripts/record_reconstruction.py'],'record-reconstruction')
 if not (video/'reconstruction_reference.mp4').exists():native(['env','V2D_REPLAY_STATE_FILE=/episode/evaluation/reconstruction-reference-states.npz','V2D_STATE_OUTPUT=/episode/evaluation/reconstruction-rendered-states.npz','python','-u','/episode/scripts/render_source_camera.py',f'evaluation.checkpoint={checkpoint}',f"task.parquet='/episode/human_motion_data/track3_video/processed/sequence_id={c['sequence']}/robot_name=sharpa_wave'",*shared,'viewer.output_path=/episode/videos/reconstruction_reference.mp4','hydra.run.dir=/episode/hydra/reconstruction-reference'],'reconstruction-reference')
 # The original RGB lens convention is restored for fair side-by-side playback.
 cal=json.loads((BASE/'datasets/video_to_data_challenge/track_3/public/meta/camera_calibration.json').read_text())['cameras']['ego_cam_a'];ww,hh=cal['resolution'];K=np.array(cal['intrinsic_matrix']);D=np.array(cal['distortion_coefficients']);newK,_=cv2.getOptimalNewCameraMatrix(K,D,(ww,hh),.7,(ww,hh));yy,xx=np.indices((768,1024),dtype=np.float32);raw=np.stack([ww-1-((xx+.5)*ww/1024-.5),hh-1-((yy+.5)*hh/768-.5)],axis=-1);und=cv2.undistortPoints(raw.reshape(-1,1,2),K,D,P=newK).reshape(768,1024,2);mx=((ww-1-und[:,:,0])*1024/ww).astype(np.float32);my=((hh-1-und[:,:,1])*768/hh).astype(np.float32)
 source=cv2.VideoCapture(c['source']);policy=cv2.VideoCapture(str(video/'trained_policy_source_camera.mp4'));reconstruction=cv2.VideoCapture(str(video/'reconstruction_reference.mp4'));assert all(cap.isOpened() for cap in [source,policy,reconstruction]);spec=[('original_vs_policy',1280,480),('original_vs_reconstruction',1280,480),('policy_rollout_source_timed',1024,768)];writers={n:cv2.VideoWriter(str(video/f'{n}.temporary.mp4'),cv2.VideoWriter_fourcc(*'mp4v'),20,(ww,hh)) for n,ww,hh in spec};assert all(x.isOpened() for x in writers.values());index=-1;examples=[]
 for i in range(T):
  ok,original=source.read();assert ok
  target=int(np.rint(i*(N-1)/(T-1)))
  for j in range(target-index):
   ok,physical=policy.read();assert ok
   ok,reference=reconstruction.read();assert ok
  index=target;physical=cv2.remap(physical,mx,my,cv2.INTER_LINEAR);reference=cv2.remap(reference,mx,my,cv2.INTER_LINEAR)
  writers['policy_rollout_source_timed'].write(physical)
  panels=[cv2.resize(im,(640,480)) for im in [original,physical,reference]]
  labels=[f'Original human | frame {i} | {i/20:.2f}s',f'Trained policy | source timing | {speed:g}x physics','Reconstruction reference | kinematic preview']
  for im,label in zip(panels,labels):cv2.rectangle(im,(0,0),(640,36),(20,20,20),-1);cv2.putText(im,label,(10,25),0,.55,(255,255,255),1,cv2.LINE_AA)
  a=np.concatenate([panels[0],panels[1]],axis=1);writers['original_vs_policy'].write(a);writers['original_vs_reconstruction'].write(np.concatenate([panels[0],panels[2]],axis=1))
  if i in set(np.linspace(0,T-1,12).round().astype(int)):examples.append(cv2.resize(a,(640,240)))
 for cap in [source,policy,reconstruction]:cap.release()
 for writer in writers.values():writer.release()
 for name,_,_ in spec:
  temporary=video/f'{name}.temporary.mp4';subprocess.run(['ffmpeg','-hide_banner','-loglevel','error','-y','-i',str(temporary),'-c:v','libx264','-preset','fast','-crf','20','-pix_fmt','yuv420p','-movflags','+faststart',str(video/f'{name}.mp4')],check=True);temporary.unlink()
 cv2.imwrite(str(evaluation/'rollout-review.jpg'),np.concatenate(examples))
 checks=[]
 for name,expected in [('original_vs_policy',T),('original_vs_reconstruction',T),('policy_rollout_source_timed',T),('trained_policy_source_camera',N),('trained_policy_rollout',N),('reconstruction_reference',N)]:
  cap=cv2.VideoCapture(str(video/f'{name}.mp4'));fps=cap.get(cv2.CAP_PROP_FPS);count=0
  while cap.read()[0]:count+=1
  cap.release();assert count==expected and abs(fps-20)<.01,(name,count,expected,fps);checks.append({'video':name+'.mp4','frames':count,'fps':fps})
 metrics=json.loads((evaluation/'final-metrics.json').read_text());fidelity_path=(r/chosen['source_fidelity_report']) if chosen.get('source_fidelity_report') else evaluation/(Path(chosen['recorded_states']).stem.removesuffix('-states')+'-source-fidelity.json');fidelity=json.loads(fidelity_path.read_text());reference=json.loads((evaluation/'reference-source-fidelity.json').read_text());low={n:sum(f['objects'][n]['iou']<.5 for f in fidelity['frames']) for n in c['objects']};quality={'all_internal_frame_checks_passed':all(v==0 for v in low.values()) and metrics['metrics']['chord_sr']==1. and chosen['single_attempt_tracking']['passed_all_frames'],'full_human_video_match_verified':False,'match_interpretation':'Internal thresholds and inferred poses do not establish exact human-video or robot-hand pixel equivalence.','source_frames_below_internal_0_5_iou':low,'completion_rate_against_video_estimate':metrics['metrics']['chord_sr'],'pipeline_completion_does_not_imply_successful_imitation':True}
 report={'episode':c['sequence'],'objects':c['objects'],'source_frames':T,'all_original_frames_visually_reviewed':c['all_frames_visually_reviewed'],'dataset_audit':'../evaluation_batch_20261015/dataset-resource-audit.json','source_review':c,'reconstruction_source_fidelity':reference['aggregate'],'observed_rest_stability':json.loads((r/'reconstruction/observed-rest-stability.json').read_text()) if (r/'reconstruction/observed-rest-stability.json').exists() else None,'object_contact_projection':json.loads((r/'research/native-object-contact-projection.json').read_text()) if (r/'research/native-object-contact-projection.json').exists() else None,'camera_stability':json.loads((r/'reconstruction/camera-stability.json').read_text()),'pose_continuity':json.loads((r/'reconstruction/pose-continuity.json').read_text()),'collision_reference_refinement':json.loads((r/'research/collision-reference-selection.json').read_text()) if (r/'research/collision-reference-selection.json').exists() else None,'training':json.loads((r/'coherent-training-provenance.json').read_text()),'selection':selection,'native_evaluation':metrics,'selected_attempt_reference_tracking':chosen['single_attempt_tracking'],'selected_source_fidelity':fidelity['aggregate'],'quality_assessment':quality,'validation':checks,'protocol':{'object_assistance':0.,'failure_resets':False,'source_time_mapping':f'round(i*{N-1}/{T-1}); uniform native sampling; no nonlinear time warping','physics_motion_speed':speed,'physics_substeps':physics_substeps,'physics_integration_hz':20*physics_substeps,'comparison_video_duration_s':T/20,'both_camera_views_use_identical_recorded_physical_states':True},'limitations':['Source masks, hands and metric depth are estimates; confidential evaluation poses were not used.','Training and Newton dynamics do not guarantee an exact match to every human motion.',('Native simulation preserves original source frame indices and timing.' if c.get('validated_video_pipeline') else 'Legacy native physics uses half speed; comparison playback follows source timing.'),'Robot morphology differs from the human hands; full laboratory and human arms are not rebuilt.']}
 training_lines=(r/'flash_chord_coherent_actor_transfer/metrics.jsonl').read_text().splitlines()
 if training_lines:report['training']['actual_additional_environment_steps']=json.loads(training_lines[-1])['additional_environment_steps']
 stop=r/'research/training-early-stop.json'
 if stop.exists():report['training']['validated_early_stopping']=json.loads(stop.read_text())
 (r/'result-report.json').write_text(json.dumps(report,indent=2));d=r/'deliverables';d.mkdir(exist_ok=True)
 artifacts={'original_vs_policy.mp4':video/'original_vs_policy.mp4','original_vs_reconstruction.mp4':video/'original_vs_reconstruction.mp4','rollout_source_timed.mp4':video/'policy_rollout_source_timed.mp4','rollout_3d.mp4':video/'trained_policy_rollout.mp4','scene.glb':r/'reconstruction/rebuilt_scene.glb','policy.safetensors':r/checkpoint.removeprefix('/episode/'),'training_state.safetensors':r/checkpoint.removeprefix('/episode/').replace('policy_','state_'),'policy_states.npz':state,'report.json':r/'result-report.json'}
 artifacts.update({name:video/name for name in ['trained_policy_source_camera.mp4','reconstruction_reference.mp4']})
 for name,target in artifacts.items():
  p=d/name
  if p.is_symlink():p.unlink()
  p.symlink_to(Path('..')/target.relative_to(r))
 manifest={name:{'path':str(path.relative_to(r)),'bytes':path.stat().st_size,'sha256':hashlib.sha256(path.read_bytes()).hexdigest()} for name,path in artifacts.items()};(d/'manifest.json').write_text(json.dumps(manifest,indent=2))
 # Retain the selected policy and resumable state. Candidate metrics, recorded
 # physical attempts and hashes remain available after releasing bulky weights.
 selected_name=Path(checkpoint).name;selected_step=selected_name.removeprefix('policy_').removesuffix('.safetensors');removed=[]
 retained={selected_name,f'state_{selected_step}.safetensors'}
 if c.get('experimental_training'):
  retained.update(p.name for p in (r/'flash_chord_coherent_actor_transfer').glob('*.safetensors'))
 if c.get('validated_video_pipeline'):
  resumable=[p for p in (r/'flash_chord_coherent_actor_transfer').glob('state_*.safetensors') if p.with_name(p.name.replace('state_','policy_')).exists()]
  if resumable:
   latest=max(resumable,key=lambda p:int(p.stem.split('_')[-1]));retained.update([latest.name,latest.name.replace('state_','policy_')])
 for candidate in (r/'flash_chord_coherent_actor_transfer').glob('*.safetensors'):
  if candidate.name in retained:continue
  removed.append({'name':candidate.name,'bytes':candidate.stat().st_size,'sha256':hashlib.sha256(candidate.read_bytes()).hexdigest()});candidate.unlink()
 (r/'research/checkpoint-retention.json').write_text(json.dumps({'retained_step':int(selected_step),'selected_resumable_state_retained':True,'removed_reviewed_candidates':removed},indent=2))
 print(ep,'exported',quality,flush=True)
if __name__=='__main__':
 for ep in map(int,sys.argv[1:]):export(ep)
