"""Reclaim regenerable compiler caches from stopped retired scan-based runtimes."""
from pathlib import Path
from datetime import datetime,timezone
import fcntl,inspect,json,os,re,shlex,shutil,subprocess,time
from improvement_common import BASE,save,utc


def read(path):
    try:return json.loads(Path(path).read_text())
    except (OSError,ValueError):return {}


def clear_compiler_caches(root='/'):
    root=Path(root).resolve();rows=[]
    for suffix in ['root/.cache/warp','root/.nv/ComputeCache']:
        path=root/suffix
        if not path.exists():continue
        assert not path.is_symlink() and path.resolve()==path,'Cache path is redirected'
        size=sum(p.lstat().st_size for p in path.rglob('*') if p.is_file() and not p.is_symlink())
        shutil.rmtree(path);rows.append(dict(path=str(path),removed_bytes=size))
    return rows


def docker(*args):
    result=subprocess.run(['sg','docker','-c',shlex.join(['docker',*args])],capture_output=True,text=True,check=True,timeout=5)
    return result.stdout


def free_bytes():
    disk=os.statvfs(BASE);return disk.f_bavail*disk.f_frsize


def retired(details):
    if details.get('State',{}).get('Status')!='exited':return False
    config=details.get('Config',{})
    if config.get('Entrypoint')!=['/bin/sleep'] or config.get('Cmd')!=['infinity']:return False
    mounts=details.get('Mounts',[])
    if any(m['Destination'].startswith('/root') for m in mounts):return False
    source=next((m['Source'] for m in mounts if m['Destination']=='/episode'),None)
    if not source:return False
    root=Path(source).resolve()
    if not root.is_relative_to((BASE/'runs').resolve()):return False
    cfg=root/'episode-config.json'
    if not cfg.is_file() or read(cfg).get('own_geometry_required'):return False
    try:
        stamp=re.sub(r'(\.\d{6})\d+',r'\1',details['State']['FinishedAt'])
        age=datetime.now(timezone.utc).timestamp()-datetime.fromisoformat(stamp.replace('Z','+00:00')).timestamp()
    except (KeyError,ValueError,TypeError):return False
    return age>=86400


def stopped_candidates(listing,processed):
    candidates=[]
    for line in listing.splitlines():
        cid,name=line.split(maxsplit=1)
        if cid[:12] not in processed and re.fullmatch(r'v2d-.*newton(?:-legacy-\d+)?',name):candidates.append((cid,name))
    for start in range(0,len(candidates),16):
        batch=candidates[start:start+16]
        details=json.loads(docker('inspect',*[cid for cid,name in batch]))
        for (cid,name),detail in zip(batch,details):yield cid,name,detail


def reclaim(out,target_bytes=12_000_000_000,maximum_containers=3):
    out=Path(out);path=out/'storage-cache-maintenance.json';actions=[]
    if free_bytes()>=target_bytes or read(out.parent/'user-pause.json').get('status')=='paused_by_user':return actions
    with (out/'storage-cache-maintenance.lock').open('a') as lock:
        try:fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
        except BlockingIOError:return actions
        history=read(path);history.setdefault('containers',{})
        old=read(out/'diagnostics/storage-cache-cleanup-20261103.json')
        processed={r['id'][:12] for r in old.get('containers',[]) if r.get('final_state')=='exited'}|set(history['containers'])
        started=time.monotonic();before=free_bytes()
        script='import json,shutil\nfrom pathlib import Path\n'+inspect.getsource(clear_compiler_caches)+'\nprint(json.dumps(clear_compiler_caches()))'
        listing=docker('ps','-a','--filter','status=exited','--format','{{.ID}} {{.Names}}')
        for cid,name,details in stopped_candidates(listing,processed):
            if (time.monotonic()-started>12 or len(actions)>=maximum_containers or free_bytes()>=target_bytes
                    or read(out.parent/'user-pause.json').get('status')=='paused_by_user'):break
            if not retired(details):continue
            row=dict(container=name,id=cid,started_utc=utc(),free_bytes_before=free_bytes())
            docker('start',cid)
            try:row['removed_caches']=json.loads(docker('exec',cid,'env','CUDA_VISIBLE_DEVICES=','JAX_PLATFORMS=cpu','python','-c',script))
            finally:docker('stop','--time','1',cid)
            row.update(final_state='exited',free_bytes_after=free_bytes());actions.append(row)
            history['containers'][cid[:12]]=row
            history.update(updated_utc=utc(),free_bytes=free_bytes(),target_bytes=target_bytes,host_artifacts_and_runtimes_retained=True)
            save(path,history)
        history.update(updated_utc=utc(),free_bytes=free_bytes(),last_reclaimed_bytes=free_bytes()-before)
        save(path,history)
    return actions
