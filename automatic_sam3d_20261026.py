"""Authenticated, resumable SAM3D geometry queue. No silent baseline fallback."""
import fcntl,json,os,shlex,subprocess,time,traceback
from pathlib import Path
from improvement_common import BASE,BATCH,save,utc,digest

OUT=BATCH/'sam3d_rebuild_20261026'
PY=BASE/'.venv/bin/python'
WEIGHTS=BASE/'video_to_data/reconstruction/data/weights/sam3d'

def status(state,**extra):
    data={'status':state,'updated_utc':utc(),'pid':os.getpid(),'method':'SAM 3D Objects',
          'report':'/artifacts/evaluation_batch_20261015/sam3d_rebuild_20261026/status.json',
          'policy_training_started':False,'exact_human_match_verified':False,**extra}
    save(OUT/'status.json',data)

def main():
    manifest=json.loads((OUT/'input-manifest.json').read_text());jobs=manifest['jobs']
    while not (OUT/'checkpoint-download-completed.json').exists():
        status('waiting_for_huggingface_access',episode_count=len(manifest['episodes']),geometry_jobs=len(jobs),
               reason='Approved Hugging Face account and local hf auth login required. HTTP 401 without credentials.')
        check="from huggingface_hub import get_token,get_hf_file_metadata,hf_hub_url; assert get_token(), 'No local HF login'; get_hf_file_metadata(hf_hub_url('facebook/sam-3d-objects','checkpoints/pipeline.yaml'))"
        with (OUT/'access-check.log').open('w') as log:
            result=subprocess.run([str(PY),'-c',check],stdout=log,stderr=subprocess.STDOUT)
        if result.returncode:time.sleep(30);continue
        status('downloading_checkpoints',episode_count=len(manifest['episodes']),geometry_jobs=len(jobs))
        code="from huggingface_hub import snapshot_download; snapshot_download('facebook/sam-3d-objects',allow_patterns=['checkpoints/*'],local_dir="+repr(str(WEIGHTS/'hf-download'))+",max_workers=2)"
        with (OUT/'checkpoint-download.log').open('a') as log:
            result=subprocess.run([str(PY),'-c',code],stdout=log,stderr=subprocess.STDOUT)
        if result.returncode:time.sleep(30);continue
        save(OUT/'checkpoint-download-completed.json',{'completed_utc':utc(),'repository':'facebook/sam-3d-objects'})
    for index,job in enumerate(jobs):
        assert digest(job['image'])==job['image_sha256'],'Source image changed after input preparation'
        assert digest(job['mask'])==job['mask_sha256'],'Object mask changed after input preparation'
        marker=Path(job['mesh']).parent/'sam3d-completed.json'
        if marker.exists() and all(Path(job[k]).is_file() for k in ['mesh','transform','intrinsics']):continue
        while True:
            free=int(subprocess.check_output(['nvidia-smi','--query-gpu=memory.free','--format=csv,noheader,nounits'],text=True).splitlines()[0])
            if free>=30000:break
            status('waiting_for_exclusive_gpu',completed_geometry_jobs=index,geometry_jobs=len(jobs),episode=job['episode'],
                   reason='The current learner may finish; SAM3D requires almost all 32GB VRAM.');time.sleep(30)
        status('reconstructing',episode=job['episode'],object=job['object'],source_frame=job['frame'],completed_geometry_jobs=index,geometry_jobs=len(jobs))
        cmd=[str(PY),'-m','v2d.sam3d.docker.run_image_to_mesh','--image_path',job['image'],'--mask_path',job['mask'],
             '--mesh_path',job['mesh'],'--transform_path',job['transform'],'--intrinsics_path',job['intrinsics'],
             '--weights_dir',str(WEIGHTS),'--seed',str(job['seed'])]
        if job.get('depth'):
            cmd+=['--depth_path',job['depth'],'--depth_intrinsics_path',job['source_camera_intrinsics'],'--depth_mask_path',job['mask']]
        env=os.environ.copy();env['PYTHONPATH']=str(BASE)
        with Path(job['mesh']).with_suffix('.log').open('w') as log:
            result=subprocess.run(['sg','docker','-c',shlex.join(cmd)],env=env,stdout=log,stderr=subprocess.STDOUT)
        if result.returncode:
            status('reconstruction_failed',episode=job['episode'],object=job['object'],source_frame=job['frame'],log=str(Path(job['mesh']).with_suffix('.log')))
            raise RuntimeError(f"SAM3D inference failed: episode {job['episode']}, {job['object']}")
        assert all(Path(job[k]).is_file() for k in ['mesh','transform','intrinsics'])
        save(marker,{'completed_utc':utc(),'method':'SAM 3D Objects','input':job,'motion_validated':False})
    # Image-to-mesh does not estimate a full motion trajectory or train a policy.
    status('geometry_ready_for_alignment_and_motion_validation',completed_geometry_jobs=len(jobs),geometry_jobs=len(jobs),episode_count=20,
           reason='Generated meshes must be aligned to metric camera/public geometry, tracked through every source frame and validated before fresh policy training.')

if __name__=='__main__':
    OUT.mkdir(parents=True,exist_ok=True)
    with (OUT/'queue.lock').open('a') as lock:
        fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
        try:main()
        except Exception:
            save(OUT/'worker-error.json',{'updated_utc':utc(),'error':traceback.format_exc()});raise
