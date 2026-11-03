"""Lossless compression of older, newly generated optimizer snapshots only."""
import gzip,hashlib,json,shutil
from pathlib import Path
from improvement_common import digest,save,utc


def archive(root):
    root=Path(root);cfg=json.loads((root/'episode-config.json').read_text())
    if not cfg.get('quality_training'):return []
    folder=root/'flash_chord_coherent_actor_transfer'
    states=sorted([p for p in folder.glob('state_*.safetensors') if p.with_name(p.name.replace('state_','policy_')).exists()],key=lambda p:int(p.stem.split('_')[-1]))
    rows=[]
    for path in states[:-1]:
        expected=digest(path);target=folder/'optimizer-archive'/(path.name+'.gz');target.parent.mkdir(exist_ok=True)
        if not target.exists():
            temporary=target.with_suffix('.tmp')
            with path.open('rb') as source,gzip.open(temporary,'wb',compresslevel=3) as destination:shutil.copyfileobj(source,destination,1024*1024)
            temporary.replace(target)
        actual=hashlib.sha256()
        with gzip.open(target,'rb') as stream:
            while block:=stream.read(1024*1024):actual.update(block)
        assert actual.hexdigest()==expected,'Optimizer archive is not an exact lossless copy'
        row=dict(original_name=path.name,original_sha256=expected,archive=str(target),archive_sha256=digest(target),policy_checkpoint_retained=True)
        save(target.with_suffix('.json'),dict(row,updated_utc=utc()));path.unlink();rows.append(row)
    return rows
