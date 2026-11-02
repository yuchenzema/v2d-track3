"""Prepare and run an isolated fresh policy experiment; never overwrite the baseline."""
from pathlib import Path
import concurrent.futures
import fcntl
import json
import os
import re
import shlex
import shutil
import stat
import subprocess
import threading
import time
from improvement_common import BASE,OUT,root,save,utc,preserve,verify_preserved,better,baseline_ready
from improvement_common import baseline as baseline_selection
from batch_native import docker,launch_training
from batch_review import monitor
from batch_export import export
from batch_publish_runtime import publish
from batch_final_audit import audit


def prepare(ep,variant=None):
    if variant and re.fullmatch(r'[a-z_]+_20261021',variant):
        assert ep in [30,32]
        r=BASE/f'runs/episode_{ep:06d}_newton_{variant}'
        assert json.loads((r/'research/preparation-status.json').read_text())['status']=='prepared_and_configuration_validated'
        cfg=json.loads((r/'episode-config.json').read_text())
        assert cfg['training_curriculum']=='contact_finetune'
        return r,cfg['native_container']
    if variant=='stable_contact_20261020':
        assert ep==10
        r=BASE/f'runs/episode_{ep:06d}_newton_{variant}'
        assert json.loads((r/'research/preparation-status.json').read_text())['status']=='prepared_and_configuration_validated'
        cfg=json.loads((r/'episode-config.json').read_text())
        assert cfg['training_numeric_failure_guard'] and cfg['physics_substeps']==20
        return r,cfg['native_container']
    if variant=='stationary_hold_20261020':
        assert ep==16
        r=BASE/f'runs/episode_{ep:06d}_newton_{variant}'
        prepared=json.loads((r/'research/preparation-status.json').read_text())
        assert prepared['status']=='prepared_and_configuration_validated'
        cfg=json.loads((r/'episode-config.json').read_text())
        assert cfg['training_reference']=='track3_stationary_spoon_hold_probe' and cfg['training_curriculum']=='contact_finetune'
        return r,cfg['native_container']
    if variant not in [None,'grasp_curriculum_20261020','backward_grasp_20261020']:raise ValueError('Unknown independent experiment variant')
    baseline=root(ep);r=BASE/'runs'/f'episode_{ep:06d}_newton_{variant or "improvement_20261019"}'
    container=(f'v2d-backward-episode{ep}-newton' if variant=='backward_grasp_20261020' else
               f'v2d-grasp-episode{ep}-newton' if variant else f'v2d-improvement-episode{ep}-newton')
    r.mkdir(exist_ok=True)
    for name in ['logs','research','evaluation','videos','deliverables','flash_chord_coherent_actor_transfer']:
        (r/name).mkdir(exist_ok=True)
    cfg=json.loads((baseline/'episode-config.json').read_text())
    cfg.update(run=str(r),baseline_run=str(baseline),training_initial_counter=0,training_world_count=1024,
               native_container=container,
               improvement_experiment='Opening-grasp pretraining followed by full-motion learning' if variant else 'Independent fresh actor with zero initial residual mean; baseline inputs preserved')
    if variant:cfg.update(training_variant=variant,training_budget=65_001_472)
    decision=r/'research/physical-grasp-reference-selection.json'
    if variant and decision.exists():
        refined=json.loads(decision.read_text())
        if refined.get('approved_for_independent_training'):
            assert (r/'human_motion_data'/refined['reference']).is_dir()
            cfg.update(training_reference=refined['reference'],wrist_motor_profile=refined['wrist_motor_profile'],
                physical_grasp_refinement=str(decision.relative_to(r)))
            if refined.get('training_curriculum')=='backward':
                cfg.update(training_curriculum='backward',training_budget=70_001_664)
                if refined.get('backward_hold_source_frame') is not None:
                    cfg['backward_hold_source_frame']=int(refined['backward_hold_source_frame'])
                if refined.get('wrist_side_motor_profiles'):
                    cfg['wrist_side_motor_profiles']=refined['wrist_side_motor_profiles']
                if refined.get('strict_reset_retrain_prefix_steps'):
                    cfg.update(strict_reset_retrain_prefix_steps=int(refined['strict_reset_retrain_prefix_steps']),
                               training_budget=70_001_664+int(refined['strict_reset_retrain_prefix_steps']))
                if refined.get('adaptive_single_prefix'):
                    cfg['adaptive_single_prefix']=True
            if refined.get('training_curriculum')=='contact_finetune':
                cfg.update(training_curriculum='contact_finetune',training_budget=50_000_896,
                    training_initial_counter=int(refined.get('training_initial_counter',50_000_896)),training_world_count=512,
                    improvement_experiment='Video-calibrated contact reconstruction and episode-actor finetuning')
                cfg['wrist_side_motor_profiles']=refined.get('wrist_side_motor_profiles',{})
                for key,value in refined.get('native_contact_capacity',{}).items():
                    assert key in ['nconmax','njmax'] and 512<=int(value)<=8192
                    cfg['physics_'+key]=int(value)
    # These are actual copies, so no training helper can mutate a baseline via a hard link.
    for name in ['reconstruction','scripts']:
        if not (r/name).exists():shutil.copytree(baseline/name,r/name)
    for reference in set([cfg['training_reference'],'track3_video']):
        target=r/'human_motion_data'/reference
        if not target.exists():target.parent.mkdir(exist_ok=True);shutil.copytree(baseline/'human_motion_data'/reference,target)
    for name in ['source-video-review.json','collision-reference-selection.json']:
        source=baseline/'research'/name
        if source.exists() and not (r/'research'/name).exists():shutil.copy2(source,r/'research'/name)
    # The export reports compare the independent policy with this unchanged
    # source-fit reference; it is not a policy evaluation or a trainable input.
    for name in ['reference-source-fidelity.json','physical-reference-source-fidelity.json']:
        source=baseline/'evaluation'/name
        if source.exists() and not (r/'evaluation'/name).exists():shutil.copy2(source,r/'evaluation'/name)
    link=r/'.venv-video'
    if not link.exists():link.symlink_to(BASE/'runs/episode_000005_newton_20261015/.venv-video',target_is_directory=True)
    save(r/'episode-config.json',cfg)
    script=r/'scripts/train_coherent_actor_transfer.py'
    if not (r/'coherent-input-checksums.json').exists():
        training_script='contact_finetune_training.py' if cfg.get('training_curriculum')=='contact_finetune' else 'backward_grasp_training.py' if cfg.get('training_curriculum')=='backward' else 'grasp_curriculum_training.py' if variant else 'fresh_improvement_training.py'
        shutil.copy2(BASE/training_script,script)
    if cfg.get('training_curriculum') in ['backward','contact_finetune']:
        for helper in ['validation_stop.py','training_finger_reset.py']:shutil.copy2(BASE/helper,r/'scripts'/helper)
    name=container
    check=subprocess.run(['sg','docker','-c',shlex.join(['docker','inspect',name])],stdout=subprocess.DEVNULL,stderr=subprocess.DEVNULL)
    if check.returncode:
        image=subprocess.check_output(['sg','docker','-c',shlex.join(['docker','inspect',f'v2d-batch-episode{ep}-newton','--format','{{.Image}}'])],text=True).strip()
        command=['run','-d','--gpus','all','--runtime','nvidia','--ipc','host','--name',name,
                 '-v',f'{r}:/episode','-v',f'{BASE}/video_to_data/robotic_grounding/flash_chord:/workspace',
                 '-v',f'{BASE}/video_to_data/robotic_grounding/source/robotic_grounding/robotic_grounding/assets:/v2d/assets:ro',
                 '-v',f'{BASE}/runs/episode_000001_newton_v2_20261014/flash_chord:/prior:ro',
                 '-e','PYGLET_HEADLESS=true','-e','XLA_PYTHON_CLIENT_PREALLOCATE=false',
                 '--workdir','/workspace','--entrypoint','/bin/sleep']
        for node in sorted([*Path('/dev').glob('nvidia*'),*Path('/dev/nvidia-caps').glob('*')]):
            if stat.S_ISCHR(node.stat().st_mode):command+=['--device',str(node)]
        command += [image,'infinity'];docker(command,r/'logs/create-newton.log')
    return r,name


def validate_configuration(ep):
    r,name=prepare(ep)
    docker(['exec',name,'env','CUDA_VISIBLE_DEVICES=','JAX_PLATFORMS=cpu','V2D_VALIDATE_CONFIG_ONLY=1',
            'python','-u','/episode/scripts/train_coherent_actor_transfer.py'],r/'logs/configuration-validation.log')
    config=json.loads((r/'coherent-training-config.json').read_text())
    assert not config['training']['resume']
    assert config['training']['network']['residual_mean_head']=='zero_dense'
    assert config['training']['total_environment_steps']==50_000_896
    print(ep,'fresh experiment config/inputs validated without policy training',flush=True)
    return r,name


def run(ep,variant=None):
    assert baseline_ready(),'Finish and verify every baseline episode before beginning improvement training'
    preserve(ep);r,name=prepare(ep,variant);status=r/'research/improvement-status.json'
    result_directory=OUT.parent/('improvements_20261020_backward' if variant=='backward_grasp_20261020' else 'improvements_20261020_grasp') if variant else OUT
    if variant and variant.endswith('_20261021'):
        result_directory=OUT.parent/'improvements_20261021'
    result_file=result_directory/(f'episode_{ep:06d}-{variant}-result.json' if variant and variant.endswith('_20261021') else f'episode_{ep:06d}-fresh-result.json')
    result_directory.mkdir(exist_ok=True)
    experiment='Opening-grasp pretraining followed by full-motion FlashSAC learning' if variant else 'Independent fresh FlashSAC actor with zero initial residual mean'
    if json.loads((r/'episode-config.json').read_text()).get('training_curriculum')=='contact_finetune':
        experiment='Video-calibrated contact reconstruction and episode-actor finetuning'
    try:
        save(status,{'episode':ep,'status':'training','updated_utc':utc()})
        save(result_file,{'episode':ep,'status':'training','updated_utc':utc(),
             'run':str(r),'experiment':experiment})
        if not (r/'flash_chord_coherent_actor_transfer/TRAINING_COMPLETED').exists():
            training_failed=threading.Event()
            def train():
                try:
                    with (r/'flash_chord_coherent_actor_transfer/training-process.lock').open('a') as lock:
                        fcntl.flock(lock,fcntl.LOCK_EX)
                        while not (r/'flash_chord_coherent_actor_transfer/TRAINING_COMPLETED').exists():
                            # Docker requires a PID column even when only args are inspected.
                            top=subprocess.run(['sg','docker','-c',shlex.join(['docker','top',name,'-eo','pid,args'])],capture_output=True,text=True,check=True).stdout
                            if '/episode/scripts/train_coherent_actor_transfer.py' not in top:
                                launch_training(name,r);return
                            # Reattach after an orchestrator restart; an existing learner owns its checkpoint stream.
                            time.sleep(15)
                except BaseException:
                    training_failed.set()
                    raise
            with concurrent.futures.ThreadPoolExecutor(max_workers=2) as pool:
                trainer=pool.submit(train)
                reviewer=pool.submit(monitor,ep,root_override=r,container_override=name,stop_event=training_failed)
                trainer.result();reviewer.result()
        elif not (r/'final-controller-selection.json').exists():
            monitor(ep,root_override=r,container_override=name)
        save(status,{'episode':ep,'status':'rendering','updated_utc':utc()})
        export(ep,root_override=r,container_override=name)
        # Runtime publisher normally waits for the final report to settle.
        while time.time()-(r/'result-report.json').stat().st_mtime<11:time.sleep(1)
        publish(r)
        validation=audit(r)
        if not validation['integrity_passed']:raise RuntimeError(validation['issues'])
        selected=json.loads((r/'final-controller-selection.json').read_text())['selected']
        original=baseline_selection(ep)
        old_report=json.loads((root(ep)/'result-report.json').read_text())
        new_report=json.loads((r/'result-report.json').read_text())
        old_sr=old_report['quality_assessment']['completion_rate_against_video_estimate']
        new_sr=new_report['quality_assessment']['completion_rate_against_video_estimate']
        improved=better(selected,original) and new_sr>=old_sr-.0625
        verify_preserved(ep)
        result={'episode':ep,'status':'completed','updated_utc':utc(),'run':str(r),
                'experiment':experiment,
                'baseline':original,'candidate':selected,'baseline_128_world_completion':old_sr,
                'candidate_128_world_completion':new_sr,'measured_improvement':improved,
                'baseline_artifacts_preserved_and_hash_verified':True,'artifact_integrity':validation,
                'full_human_motion_match_verified':False}
        save(status,result);save(result_file,result)
        print(ep,'fresh experiment complete; measured improvement',improved,flush=True)
        return result
    except Exception as error:
        import traceback
        result={'episode':ep,'status':'failed','updated_utc':utc(),'run':str(r),'error':str(error),'traceback':traceback.format_exc()}
        save(status,result);save(result_file,result)
        raise


if __name__=='__main__':
    import sys
    for ep in map(int,sys.argv[2:]):
        validate_configuration(ep) if sys.argv[1]=='validate' else run(ep)
