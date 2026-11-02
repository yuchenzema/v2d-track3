"""Preserve training history and align new training resets with unassisted rollout resets."""
import fcntl,json,os,shlex,shutil,signal,subprocess,time
from pathlib import Path
from improvement_common import BASE,BATCH,save,utc,digest
from progress_dashboard import latest_metrics
from fresh_improvement_run import prepare

TARGETS=[(19,'grasp_curriculum_20261020'),(20,'backward_grasp_20261020'),
         (30,'grasp_curriculum_20261020'),(33,'backward_grasp_20261020')]
audit_path=BATCH/'improvements_20261020_grasp/training-reset-support-audit.json'
priority_path=BATCH/'training-priority.json'

def terminate_host(metadata,script):
    row=json.loads(metadata.read_text());proc=Path(f'/proc/{row["pid"]}/cmdline')
    if proc.exists():
        assert str(BASE/script).encode() in proc.read_bytes()
        os.kill(row['pid'],signal.SIGTERM)

def main():
    original_priority=json.loads(priority_path.read_text())
    before=BATCH/'improvements_20261020_grasp/training-reset-admission-before-audit.json'
    if not before.exists():save(before,original_priority)
    priority={**original_priority,'active':True,'updated_utc':utc(),
        'allowed_run_names':['episode_000032_newton_grasp_curriculum_20261020'],
        'additional_allowed_run_names':[],'held_for_reconstruction_review':[19,20,30,33],
        'reason':'Temporarily pause new admissions while matching reset dynamics to explicit, unassisted evaluation'}
    save(priority_path,priority)
    audit={'updated_utc':utc(),'issue':'Sampled training reset enables full VOC during a 20-step settling hold even when curriculum VOC is zero.',
        'training_reset_code':'flash_chord/lifecycle/reset.py:sample_reference_reset',
        'published_rollouts_affected_by_object_support':False,
        'published_evaluation_composition':'Explicit reset, zero force/torque, no settling preamble',
        'correction':{'reset_settling_steps':0,'object_force_limit_N':0.,'object_torque_limit_Nm':0.},
        'status':'preserving_history','episodes':{}}
    save(audit_path,audit)
    with Path('/tmp/v2d-track3-native-gpu-v2.lock').open('a') as lock:
        # Wait for current native checks to finish before detaching their host workers.
        fcntl.flock(lock,fcntl.LOCK_EX)
        for metadata,script in [(BATCH/'improvements_20261020_grasp/worker.json','grasp_curriculum_queue.py'),
                                (BATCH/'improvements_20261020_backward/worker.json','backward_policy_queue.py')]:
            terminate_host(metadata,script)
        for ep,container in [(19,'v2d-grasp-episode19-newton'),(20,'v2d-backward-episode20-newton')]:
            command=['docker','exec',container,'pkill','-TERM','-f','(^|/)(python|python3) -u /episode/scripts/train_coherent_actor_transfer.py$']
            result=subprocess.run(['sg','docker','-c',shlex.join(command)],capture_output=True,text=True)
            assert result.returncode in (0,1),(ep,result.stderr)
    print('Affected learners stopped; episode 32 continues. Preserving histories and validating strict resets.',flush=True)
    for ep,variant in TARGETS:
        r=BASE/'runs'/f'episode_{ep:06d}_newton_{variant}'
        cfg_path=r/'coherent-training-config.json'
        metrics=latest_metrics(r/'flash_chord_coherent_actor_transfer/metrics.jsonl')
        history=r/'research/supported-reset-history';history.mkdir(exist_ok=True)
        for name in ['coherent-training-config.json','coherent-training-provenance.json']:
            p=r/name
            if p.exists() and not (history/name).exists():shutil.copy2(p,history/name)
        original_config=history/'coherent-training-config.json'
        old=json.loads(original_config.read_text()) if original_config.exists() else {}
        p=r/'flash_chord_coherent_actor_transfer/metrics.jsonl'
        if p.exists() and p.stat().st_size:
            if not (history/'metrics.jsonl').exists():shutil.move(p,history/'metrics.jsonl')
        if not metrics:metrics=latest_metrics(history/'metrics.jsonl')
        retained=0
        if ep==19:
            retained=5_000_192
            marker=r/'grasp_pretraining/stage_0/TRAINING_COMPLETED'
            backup_marker=marker.with_name('TRAINING_COMPLETED_WITH_SUPPORTED_RESET')
            if marker.exists():marker.rename(backup_marker)
            else:assert backup_marker.exists()
            for prefix in ['state','policy']:
                p=r/f'grasp_pretraining/stage_0/{prefix}_{retained}.safetensors'
                assert p.exists()
                if not (history/p.name).exists():shutil.copy2(p,history/p.name)
            stage_one=r/'grasp_pretraining/stage_1'
            if stage_one.exists() and not (history/'stage_1').exists():shutil.move(stage_one,history/'stage_1')
        elif ep==20:
            assert not list((r/'grasp_pretraining').rglob('state_*.safetensors')), 'Preserve a newly finished checkpoint before restarting'
        selection_path=r/'research/physical-grasp-reference-selection.json'
        selection=json.loads(selection_path.read_text())
        selection.update(training_reset_protocol='explicit-equivalent-zero-support',reset_settling_steps=0,
            strict_reset_retrain_prefix_steps=retained,training_object_force_limit_N=0.,training_object_torque_limit_Nm=0.)
        save(selection_path,selection)
        shutil.copy2(BASE/'backward_grasp_training.py',r/'scripts/train_coherent_actor_transfer.py')
        r,container=prepare(ep,variant)
        frozen=r/'coherent-input-checksums.json'
        if frozen.exists():
            for name,checksum in json.loads(frozen.read_text()).items():assert digest(r/name)==checksum
        for index in range(4):
            args=['docker','exec',container,'env','CUDA_VISIBLE_DEVICES=','JAX_PLATFORMS=cpu','V2D_VALIDATE_CONFIG_ONLY=1',
                  'V2D_GRASP_STAGE=prefix',f'V2D_BACKWARD_STAGE={index}','python','-u','/episode/scripts/train_coherent_actor_transfer.py']
            with (r/f'logs/strict-reset-stage-{index}-configuration.log').open('w') as stream:
                subprocess.run(['sg','docker','-c',shlex.join(args)],stdout=stream,stderr=subprocess.STDOUT,check=True)
        args=['docker','exec',container,'env','CUDA_VISIBLE_DEVICES=','JAX_PLATFORMS=cpu','V2D_VALIDATE_CONFIG_ONLY=1',
              'V2D_GRASP_STAGE=full','python','-u','/episode/scripts/train_coherent_actor_transfer.py']
        with (r/'logs/strict-reset-full-configuration.log').open('w') as stream:
            subprocess.run(['sg','docker','-c',shlex.join(args)],stdout=stream,stderr=subprocess.STDOUT,check=True)
        audit['episodes'][str(ep)]={'run':str(r),'old_reset_settling_steps':old.get('env',{}).get('reset',{}).get('voc_decay_steps'),
            'old_allowed_object_support_force_N':old.get('env',{}).get('voc',{}).get('max_force'),
            'historical_environment_steps_before_restart':metrics.get('additional_environment_steps',0),
            'retained_supported_warmstart_environment_steps':retained,'training_history_preserved_at':str(history),
            'new_configuration_validated':True,'all_frozen_motion_geometry_inputs_unchanged':True}
        save(audit_path,audit);print(ep,'strict reset configuration validated',flush=True)
    audit.update(status='validated_awaiting_admission',updated_utc=utc());save(audit_path,audit)
    # Restart the independent queue. The original queue's supervisor reattaches itself.
    out=BATCH/'improvements_20261020_backward'
    with (out/'worker.log').open('a') as stream:
        process=subprocess.Popen(['/usr/bin/python3','-u',str(BASE/'backward_policy_queue.py')],cwd=BASE,
            stdin=subprocess.DEVNULL,stdout=stream,stderr=subprocess.STDOUT,start_new_session=True)
    worker=json.loads((out/'worker.json').read_text());worker.update(pid=process.pid,updated_utc=utc());save(out/'worker.json',worker)
    priority=json.loads(priority_path.read_text())
    priority.update(updated_utc=utc(),allowed_run_names=['episode_000032_newton_grasp_curriculum_20261020',
        'episode_000019_newton_grasp_curriculum_20261020','episode_000020_newton_backward_grasp_20261020'],
        additional_allowed_run_names=['episode_000020_newton_backward_grasp_20261020'],
        held_for_reconstruction_review=[30,33],
        reason='Resume validated strict-reset learners 19 and 20 first; release queued 30 and 33 after these admissions')
    save(priority_path,priority)
    print('Strict reset correction validated. Episodes 19 and 20 released first; 30 and 33 prepared and queued.',flush=True)

if __name__=='__main__':main()
