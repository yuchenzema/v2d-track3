"""Replay preserved policies with the official evaluator in the existing serial worker.

No training, reference fitting, pose editing, or completion gate changes occur here.
"""
import hashlib
import json
from pathlib import Path
import shlex
import subprocess
import time

from improvement_common import BASE,BATCH,save,utc,digest
from episode_execution_guard import episode_slot,register_episode_child


def read(path):
    try:return json.loads(Path(path).read_text())
    except (OSError,ValueError):return {}


def package_hash(runtime):
    h=hashlib.sha256()
    for name in sorted(['__init__.py','harness.py','metrics.py','rollout.py']):
        h.update(name.encode());h.update(b'\0')
        h.update((Path(runtime)/'src/flash_chord/evaluation'/name).read_bytes())
    return h.hexdigest()


def episode_root(recording):
    for parent in Path(recording).parents:
        if (parent/'episode-config.json').is_file() or ((parent/'human_motion_data').is_dir() and (parent/'scripts').is_dir()):return parent
    raise ValueError('No episode configuration for preserved recording')


def pending(out):
    request=read(out/'submission-replay-request.json')
    status=read(out/'submission-replay-status.json')
    return bool(request.get('enabled') and
                not (status.get('request_sha256')==digest(out/'submission-replay-request.json') and
                     status.get('status') in {'complete','blocked'}))


def verify_output(path,ep,checkpoint_sha,script_sha,code_sha):
    import numpy as np
    import pyarrow.parquet as pq
    table=pq.read_table(path);meta=table.schema.metadata or {}
    required={'episode_index':str(ep),'checkpoint_sha256':checkpoint_sha,
              'eval_script_sha256':script_sha,'eval_code_sha256':code_sha,
              'world_count':'1','start_frame':'0','quaternion_order':'xyzw'}
    for key,value in required.items():
        assert meta.get(('flash_chord.'+key).encode())==value.encode(),('Official provenance mismatch',key)
    assert table['environment_valid'].to_numpy().all(),'Non-finite simulation environment'
    columns=['achieved_'+n for n in ['position_x','position_y','position_z','quaternion_x','quaternion_y','quaternion_z','quaternion_w']]
    poses=np.column_stack([table[n].to_numpy() for n in columns])
    assert np.isfinite(poses).all(),'Non-finite achieved pose'
    assert np.allclose(np.linalg.norm(poses[:,3:],axis=1),1.,atol=1e-4),'Invalid XYZW quaternion'
    return dict(sha256=digest(path),rows=len(table),provenance={k.decode():v.decode() for k,v in meta.items() if k.startswith(b'flash_chord.')})


def docker_command(root,checkpoint,kit,runtime,output,ep):
    assets=BASE/'video_to_data/robotic_grounding/source/robotic_grounding/robotic_grounding/assets'
    prior=BASE/'runs/episode_000001_newton_v2_20261014/flash_chord'
    args=['run','--rm','--runtime','nvidia','--gpus','all','--ipc','host','--network','none',
          '--name',f'v2d-official-submission-episode{ep}',
          '-v',f'{root}:/episode:ro','-v',f'{checkpoint}:/checkpoint/policy.safetensors:ro',
          '-v',f'{kit}:/official-kit:ro','-v',f'{runtime}:/workspace:ro',
          '-v',f'{Path(output).parent}/runtime-cache/geometry:/workspace/src/flash_chord/assets/.cache',
          '-v',f'{Path(output).parent}/runtime-cache/user:/root/.cache',
          '-v',f'{assets}:/v2d/assets:ro','-v',f'{prior}:/prior:ro',
          '-v',f'{output}:/submission-output',
          '-e','XLA_PYTHON_CLIENT_PREALLOCATE=false','-e','PYGLET_HEADLESS=true',
          '--workdir','/workspace','--entrypoint','python','flash_chord',
          '-u','/official-kit/flash_chord/scripts/evaluate_policy.py',
          'evaluation.checkpoint=/checkpoint/policy.safetensors','evaluation.world_count=1',
          'evaluation.reset_mode=explicit','evaluation.start_frame=0','logging.mode=disabled',
          f'evaluation.metrics_output=/submission-output/episode_{ep:06d}.json',
          f'evaluation.object_trajectories_output=/submission-output/episode_{ep:06d}.parquet',
          f'hydra.run.dir=/submission-output/hydra/episode_{ep:06d}']
    return ['sg','docker','-c',shlex.join(['docker',*args])]


def pack_results(request,staged,destination,kit):
    """Use the provided packer and uploader dry-run, with no provenance edits."""
    for item in staged.values():
        for mesh in item['meshes'].values():
            assert digest(Path(mesh['submitted_mesh']))==mesh['submitted_sha256'],'Selected mesh changed'
    python=BASE/'.venv-submission/bin/python'
    target=destination/'track3_submission.parquet'
    commands=[('pack',[str(python),str(kit/'tools/pack_submission.py'),'--track','track_3',
        '--results',str(destination/'results'),'--meshes',str(Path(request['staging'])/'meshes'),
        '--code-commit',request['code_commit_url'],'-o',str(target)]),
        ('preflight',[str(python),str(kit/'tools/preflight.py'),'--track','track_3',
            '--results',str(destination/'results')]),
        ('dry-run',[str(python),str(kit/'submit/v2d_submit.py'),str(target),'--track','track_3',
            '-m','Preserved Track 3 evaluation policies; official replay','--dry-run'])]
    for label,command in commands:
        with (destination/'logs'/f'{label}.log').open('w') as log:
            subprocess.run(command,cwd=BASE,stdout=log,stderr=subprocess.STDOUT,check=True)
    return dict(file=str(target),sha256=digest(target),official_packer_used=True,
                provenance_edited=False,dry_run_passed=True,uploaded=False)


def run(out,focus):
    if not pending(out):return False
    request_path=out/'submission-replay-request.json';request=read(request_path)
    if read(BATCH/'user-pause.json').get('status')=='paused_by_user':return True
    if read(out/'controller-reload-request.json').get('pending'):return True
    assert request['episodes']==focus,'Official replay scope differs from the focused roster'
    sha=request['code_commit'];current=subprocess.check_output(['git','rev-parse','HEAD'],cwd=BASE,text=True).strip()
    assert current==sha,'Commit changed after replay was requested'
    assert not subprocess.check_output(['git','status','--porcelain','--untracked-files=no'],cwd=BASE,text=True).strip(),'Modified producing code'
    kit=Path(request['kit']);runtime=Path(request['runtime']);destination=Path(request['output'])
    results=destination/'results';logs=destination/'logs';results.mkdir(parents=True,exist_ok=True);logs.mkdir(exist_ok=True)
    # The evaluator and reference remain read-only; only derived collision and
    # kernel caches need writable storage. Keep caches across replay containers.
    (runtime/'src/flash_chord/assets/.cache').mkdir(exist_ok=True)
    for folder in ['geometry','user']:(destination/'runtime-cache'/folder).mkdir(parents=True,exist_ok=True)
    staged=read(Path(request['staging'])/'readiness.json')['episodes']
    script_sha=digest(kit/'flash_chord/scripts/evaluate_policy.py');code_sha=package_hash(runtime)
    status=read(out/'submission-replay-status.json');request_sha=digest(request_path)
    if status.get('request_sha256')!=request_sha:status={'episodes':{}}
    status.update(status='running',request_sha256=request_sha,code_commit=sha,updated_utc=utc(),
                  kind='official_replay_of_preserved_checkpoints',training_started=False,strict_completion_granted=False)
    for ep in focus:
        if read(BATCH/'user-pause.json').get('status')=='paused_by_user':return True
        if read(out/'controller-reload-request.json').get('pending'):return True
        item=staged[str(ep)];checkpoint=Path(item['checkpoint']);root=episode_root(checkpoint)
        path=results/f'episode_{ep:06d}.parquet';report=path.with_suffix('.json')
        checkpoint_sha=digest(checkpoint)
        try:
            assert checkpoint_sha==item['identity']['checkpoint_sha256'],'Preserved checkpoint changed'
            with episode_slot(root,ep):
                state=read(out/'status.json')
                state.update(status='submission_policy_replay',current_episode=ep,
                    active_optimization=dict(episode=ep,run=str(root),action='Official evaluation of preserved policy for submission',dataset_split='evaluation',submission_replay=True),
                    phase=f'Official policy evaluation for submission: episode {ep}; {len(status["episodes"])}/20 replayed',updated_utc=utc())
                save(out/'status.json',state)
                status.update(active_episode=ep,updated_utc=utc());save(out/'submission-replay-status.json',status)
                if not (path.exists() and report.exists()):
                    command=docker_command(root,checkpoint,kit,runtime,results,ep)
                    save(logs/f'episode_{ep:06d}.command.json',command)
                    from batch_gpu_gate import auxiliary_gpu
                    with auxiliary_gpu('native',minimum_free_mb=3500), (logs/f'episode_{ep:06d}.log').open('w') as log:
                        child=subprocess.Popen(command,stdout=log,stderr=subprocess.STDOUT)
                        registered=False
                        while child.poll() is None:
                            if not registered:
                                probe=subprocess.run(['sg','docker','-c',shlex.join(['docker','inspect','--format','{{.State.Pid}}',f'v2d-official-submission-episode{ep}'])],capture_output=True,text=True)
                                if probe.returncode==0 and probe.stdout.strip().isdigit():
                                    register_episode_child(int(probe.stdout.strip()));registered=True
                            state.update(updated_utc=utc());save(out/'status.json',state);time.sleep(10)
                        if child.returncode:raise subprocess.CalledProcessError(child.returncode,command)
                result=verify_output(path,ep,checkpoint_sha,script_sha,code_sha)
                result.update(checkpoint=str(checkpoint),reference_root=str(root),official_report=str(report),
                              checkpoint_training_steps=read(report).get('environment_steps'))
                status['episodes'][str(ep)]=result;save(out/'submission-replay-status.json',status)
        except Exception as error:
            status.update(status='blocked',active_episode=None,error=f'{type(error).__name__}: {error}',
                          failed_episode=ep,failure_log=str(logs/f'episode_{ep:06d}.log'),updated_utc=utc())
            save(out/'submission-replay-status.json',status)
            state=read(out/'status.json');state.update(status='submission_replay_blocked',current_episode=None,
                active_optimization=None,phase='Official submission replay failed; source repairs remain independent',updated_utc=utc())
            save(out/'status.json',state);return False
    try:
        state=read(out/'status.json');state.update(status='submission_packaging',current_episode=None,
            active_optimization=None,phase='Official replays complete; packing full roster and checking Kaggle submission',updated_utc=utc())
        save(out/'status.json',state)
        status['package']=pack_results(request,staged,destination,kit)
    except Exception as error:
        status.update(status='blocked',active_episode=None,error=f'{type(error).__name__}: {error}',
                      failure_log=str(logs),updated_utc=utc())
        save(out/'submission-replay-status.json',status);return False
    status.update(status='complete',active_episode=None,updated_utc=utc(),uploaded=False,
                  ready_for_upload=True)
    save(out/'submission-replay-status.json',status)
    state=read(out/'status.json');state.update(active_optimization=None,current_episode=None,
        status='submission_replays_complete',phase='Official full-roster Parquet packed and dry-run checked; not uploaded',updated_utc=utc())
    save(out/'status.json',state)
    return False
