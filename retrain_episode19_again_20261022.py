"""Fresh episode19 policy repeat on preserved, verified reconstructed inputs."""
import concurrent.futures
import fcntl
import time
import json
import os
from pathlib import Path
import shutil
import stat
import sys
import threading
import traceback

from improvement_common import BASE,BATCH,save,utc,digest,better
from batch_native import _docker,launch_training

SOURCE=BASE/'runs/episode_000019_newton_full_rebuild_20261022'
ROOT=BASE/'runs/episode_000019_newton_policy_repeat_20261022'
OUT=BATCH/'improvements_20261022_policy_repeat19'
NAME='v2d-rebuildrepeat22-episode19-newton'
BUDGET=120_000_000
SEED=20261023


def status(phase,state='running',**extra):
 entry={'episode':19,'pid':os.getpid(),'process_script':Path(__file__).name,'run':str(ROOT),
        'updated_utc':utc(),'status':state,'phase':phase,'description':phase,
        'activity_kind':'independent_policy_retraining',**extra}
 save(ROOT/'research/rebuild-status.json',entry)
 save(OUT/'episode_000019-status.json',entry)
 save(BATCH/'improvements_20261021/activities/episode_000019-policy-repeat-20261022.json',entry)
 save(OUT/'phase-status.json',{'phase':state if state in ['completed','completed_with_experiment_failures'] else 'fresh_policy_training',
      'updated_utc':utc(),'requested_episodes':[19],'status':entry})


def prepare():
 OUT.mkdir(exist_ok=True)
 if (ROOT/'research/retraining-input-provenance.json').exists():return
 from video_catalog import catalog
 prior=catalog(19,BASE/'runs/episode_000019_newton_batch_20261015',include_selected=True)['best']
 assert prior['run']==str(SOURCE) and prior['measured_improvement']
 assert json.loads((SOURCE/'research/rebuild-physical-preflight.json').read_text())['passed']
 ROOT.mkdir(exist_ok=True)
 for name in ['logs','research','evaluation','videos','deliverables','flash_chord_coherent_actor_transfer']:(ROOT/name).mkdir(exist_ok=True)
 for name in ['reconstruction','human_motion_data','scripts']:
  shutil.copytree(SOURCE/name,ROOT/name,symlinks=True)
 (ROOT/'.venv-video').symlink_to(SOURCE/'.venv-video',target_is_directory=True)
 (ROOT/'models').symlink_to(SOURCE/'models',target_is_directory=True)
 for name in ['source-video-review.json','rebuild-physical-preflight.json','rebuilt-release-targets.json','rebuilt-initial-support-refinement.json','rebuilt-source-constraints.json']:
  if (SOURCE/'research'/name).exists():shutil.copy2(SOURCE/'research'/name,ROOT/'research'/name)
 shutil.copytree(SOURCE/'research/full-frame-visual-review',ROOT/'research/full-frame-visual-review')
 shutil.copy2(SOURCE/'evaluation/reference-source-fidelity.json',ROOT/'evaluation/reference-source-fidelity.json')
 shutil.copy2(SOURCE/'videos/reconstruction-source-review.mp4',ROOT/'videos/reconstruction-source-review.mp4')
 cfg=json.loads((SOURCE/'episode-config.json').read_text())
 cfg.update(run=str(ROOT),baseline_run=str(SOURCE),native_container=NAME,
            training_variant='policy_repeat_20261022',training_budget=BUDGET,training_initial_counter=0,
            training_seed=SEED,improvement_experiment='Fresh policy initialization with new seed and longer training on verified reconstruction',
            reconstruction_reused_from=str(SOURCE),training_resource_reason='512 worlds; existing two-learner GPU admission queue; no interruption of active jobs',
            visual_review_method='Original source frames reviewed during the prior accepted run; source review reused here. Visual comparison of the new policy is pending.',
            new_policy_visual_comparison_completed=False)
 save(ROOT/'episode-config.json',cfg)
 checks=json.loads((SOURCE/'coherent-input-checksums.json').read_text())
 for path,expected in checks.items():assert digest(ROOT/path)==expected,path
 watched={str(SOURCE/entry['path']):entry['sha256'] for entry in json.loads((SOURCE/'deliverables/manifest.json').read_text()).values()}
 watched[str(SOURCE/'final-controller-selection.json')]=digest(SOURCE/'final-controller-selection.json')
 save(OUT/'preserved-current-episode19.json',{'release':prior,'files':watched,'verified_reconstruction_files':checks})
 save(ROOT/'research/retraining-input-provenance.json',{'updated_utc':utc(),'source_reconstruction_run':str(SOURCE),
      'geometry_camera_hand_and_object_targets_copied_without_edits':True,'all_frozen_input_hashes_match':True,
      'old_policy_copied':False,'old_optimizer_copied':False,'old_training_counter_copied':False,
      'initialization':'Fresh actor, critic, target critic, optimizer, temperature and normalizers',
      'seed':SEED,'budget':BUDGET,'training_object_assistance_zero_all_stages':True})
 script=ROOT/'scripts/train_coherent_actor_transfer.py'
 text=script.read_text().replace('reset.seed=20261041',f'reset.seed={SEED}').replace("'seed': 20261041",f"'seed': {SEED}")
 text=text.replace("'reconstruction_rerun_from_original_video': True","'reconstruction_rerun_from_original_video': False, 'reconstruction_reused_from_verified_run': episode['reconstruction_reused_from']")
 text=text.replace('source_clock_preserved=True,reconstruction_rerun_from_original_video=True','source_clock_preserved=True,reconstruction_rerun_from_original_video=False,reconstruction_reused_from_verified_run=episode[\'reconstruction_reused_from\']')
 text=text.replace("'hypothesis': 'Test negative transfer from the earlier pot/lid actor on different grasped objects'","'hypothesis': 'Test a new independent seed and longer training on fixed verified episode19 reconstruction, focusing on brush tracking through the full timeline'")
 assert 'BUDGET = int(episode[\'training_budget\'])' in text
 script.write_text(text)
 status('Preparing independent policy training',state='queued')


def container():
 import subprocess,shlex
 if subprocess.run(['sg','docker','-c',shlex.join(['docker','inspect',NAME])],stdout=subprocess.DEVNULL,stderr=subprocess.DEVNULL).returncode==0:return
 info=json.loads(_docker(['inspect','v2d-rebuild22-episode19-newton']).stdout)[0]
 args=['run','-d','--gpus','all','--runtime','nvidia','--ipc','host','--name',NAME]
 for mount in info['Mounts']:
  args+=['-v',(str(ROOT) if mount['Destination']=='/episode' else mount['Source'])+':'+mount['Destination']+('' if mount['RW'] else ':ro')]
 for device in sorted([*Path('/dev').glob('nvidia*'),*Path('/dev/nvidia-caps').glob('*')]):
  if stat.S_ISCHR(device.stat().st_mode):args+=['--device',str(device)]
 args+=['-e','PYGLET_HEADLESS=true','-e','XLA_PYTHON_CLIENT_PREALLOCATE=false','--workdir','/workspace','--entrypoint','/bin/sleep',info['Image'],'infinity']
 _docker(args,ROOT/'logs/create-newton.log')


def main():
 prepare();container()
 if not (ROOT/'coherent-input-checksums.json').exists():
  _docker(['exec',NAME,'env','CUDA_VISIBLE_DEVICES=','JAX_PLATFORMS=cpu','V2D_VALIDATE_CONFIG_ONLY=1','python','-u','/episode/scripts/run_geometry_cpu.py','/episode/scripts/train_coherent_actor_transfer.py'],ROOT/'logs/training-config-validation.log')
  cfg=json.loads((ROOT/'coherent-training-config.json').read_text())
  assert cfg['training']['total_environment_steps']==BUDGET and cfg['reset']['seed']==SEED
  assert cfg['training']['resume'] is False and cfg['training']['checkpoint_path'] is None
  assert not list((ROOT/'flash_chord_coherent_actor_transfer').glob('*.safetensors'))
 registry=BATCH/'improvements_20261021/active-runs.json';data=json.loads(registry.read_text());data['19']=ROOT.name;save(registry,data)
 if not (ROOT/'flash_chord_coherent_actor_transfer/TRAINING_COMPLETED').exists():
  status('Independent fresh FlashSAC policy training')
  save(ROOT/'research/improvement-status.json',{'episode':19,'status':'training','updated_utc':utc()})
  from batch_review import monitor
  failed=threading.Event()
  def train():
   try:launch_training(NAME,ROOT)
   except BaseException:failed.set();raise
  with concurrent.futures.ThreadPoolExecutor(max_workers=2) as pool:
   trainer=pool.submit(train);reviewer=pool.submit(monitor,19,root_override=ROOT,container_override=NAME,stop_event=failed)
   trainer.result();reviewer.result()
 status('Rendering and validating the new policy rollout')
 save(ROOT/'research/improvement-status.json',{'episode':19,'status':'rendering','updated_utc':utc()})
 from batch_export import export
 from batch_publish_runtime import publish
 from batch_final_audit import audit
 export(19,root_override=ROOT,container_override=NAME)
 # Runtime publishing waits for a stable report; exporting has just rewritten it.
 while time.time()-(ROOT/'result-report.json').stat().st_mtime<11:time.sleep(1)
 publish(ROOT)
 validation=audit(ROOT);assert validation['integrity_passed'],validation
 selected=json.loads((ROOT/'final-controller-selection.json').read_text())['selected']
 previous=json.loads((OUT/'preserved-current-episode19.json').read_text())
 baseline=previous['release']
 large=json.loads((ROOT/'evaluation/final-metrics.json').read_text())['metrics']['chord_sr']
 improved=better(selected,baseline['selected']) and large>=baseline['quality']['completion_rate_against_video_estimate']-.0625
 for name,expected in previous['files'].items():assert digest(name)==expected,('Prior accepted release changed',name)
 result={'episode':19,'status':'completed','updated_utc':utc(),'run':str(ROOT),
         'experiment':'Independent fresh120M-step episode19 policy repeat on verified reconstruction',
         'measured_improvement':improved,'baseline':baseline['selected'],'candidate':selected,
         'artifact_integrity':validation,'full_human_motion_match_verified':False}
 save(OUT/'episode_000019-fresh-result.json',result);save(ROOT/'research/improvement-status.json',result)
 status('Comparing every new rollout frame with the original human video')
 import thorough_episode_recheck as human
 from video_catalog import run_release
 human.OUT=OUT/'human_review';human.OUT.mkdir(exist_ok=True);human.__file__=__file__
 def audit_progress(completed,ep=None,frame=0,total=0,**extra):
  status(f'Comparing new episode19 rollout with original human video: {frame} / {total} frames')
  save(human.OUT/'status.json',{'updated_utc':utc(),'pid':os.getpid(),'status':'running','total_episodes':1,'current_episode':19,'current_frame':frame,'current_total_frames':total})
 human.report_status=audit_progress
 release=run_release(19,ROOT,'experiment',result)
 report=human.recheck({'episode':19,'source_frames':235,'objects':['mini_sweeper','mini_dust_pan'],'videos':{'latest':release}},[])
 report.update(new_policy_training_performed=True,audit_itself_performed_training=False,independent_fresh_training_completed=True,matches_every_human_frame_verified=False)
 save(human.OUT/'episode_000019/report.json',report)
 save(human.OUT/'status.json',{'updated_utc':utc(),'pid':os.getpid(),'status':'completed','total_episodes':1,'episodes_completed':1,'source_frames_completed':235})
 status('New policy trained and rollout rechecked; '+('verified improvement accepted' if improved else 'current accepted video retained; new candidate available for review'),state='completed',measured_improvement=improved)
 from policy_recommendations import snapshot,rounds_finished
 snapshot(finalize=rounds_finished()[0])
 import subprocess
 # Dataset staging needs pyarrow, supplied by the dedicated video environment.
 subprocess.run([str(BASE/'runs/episode_000005_newton_20261015/.venv-video/bin/python'),
                 '-c','from batch_submission_stage import update; update()'],cwd=BASE,check=True)
 error_record=OUT/'worker-error.json'
 if error_record.exists():
  record=json.loads(error_record.read_text())
  record.update(resolved=True,resolved_utc=utc(),resolution='Runtime published, integrity audit passed, all 235 source frames rechecked and staging refreshed; previous accepted policy retained.')
  save(error_record,record)


if __name__=='__main__':
 OUT.mkdir(exist_ok=True)
 with (OUT/'worker.lock').open('a') as lock:
  fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
  try:main()
  except Exception:
   error=traceback.format_exc();save(OUT/'worker-error.json',{'updated_utc':utc(),'error':error})
   save(ROOT/'research/improvement-status.json',{'episode':19,'status':'failed','updated_utc':utc(),'error':error})
   status('Policy repeat needs attention',state='completed_with_experiment_failures',error=error)
   raise
