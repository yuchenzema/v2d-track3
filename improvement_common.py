"""Shared comparison and preservation rules for the second optimization phase."""
from pathlib import Path
import hashlib
import json
import os
import time
from batch_artifacts import complete, problems

BASE = Path(__file__).resolve().parent
BATCH = BASE/'runs/evaluation_batch_20261015'
OUT = BATCH/'improvements_20261019'
EPISODES = [3,6,8,10,14,15,16,19,20,25,28,30,32,33,35,36,37,38]


def root(ep):
    return BASE/'runs'/f'episode_{ep:06d}_newton_batch_20261015'


def digest(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as f:
        for block in iter(lambda:f.read(1<<20),b''):
            h.update(block)
    return h.hexdigest()


def save(path,value):
    path=Path(path);path.parent.mkdir(parents=True,exist_ok=True)
    tmp=path.with_name(path.name+f'.{os.getpid()}.tmp');tmp.write_text(json.dumps(value,indent=2));tmp.replace(path)


def utc():
    return time.strftime('%Y-%m-%dT%H:%M:%SZ',time.gmtime())


def baseline_ready():
    p=BATCH/'batch-final-audit.json'
    return (p.exists() and json.loads(p.read_text()).get('all_requested_deliverables_complete_and_verified',False)
            and all(complete(root(ep)) for ep in EPISODES))


def baseline(ep):
    r=root(ep)
    from object_motion_quality import attach
    return attach(r,json.loads((r/'final-controller-selection.json').read_text())['selected'])


def preserve(ep):
    r=root(ep);p=OUT/'baseline_snapshots'/f'episode_{ep:06d}.json'
    manifest=json.loads((r/'deliverables/manifest.json').read_text())
    watched={str(r/entry['path']):entry['sha256'] for entry in manifest.values()}
    watched.update({str(r/name):sha for name,sha in json.loads((r/'coherent-input-checksums.json').read_text()).items()})
    watched[str(r/'final-controller-selection.json')]=digest(r/'final-controller-selection.json')
    data={'episode':ep,'baseline_run':str(r),'files':watched,'preserved_utc':utc()}
    if p.exists():
        previous=json.loads(p.read_text())
        if previous['files']!=watched:raise RuntimeError(f'Baseline episode {ep} changed since improvement began')
    else:
        issues=problems(r,verify_hashes=True)
        if issues:raise RuntimeError(f'Baseline integrity failed before improvement: {issues}')
        save(p,data)
    return data


def verify_preserved(ep):
    p=OUT/'baseline_snapshots'/f'episode_{ep:06d}.json'
    for name,sha in json.loads(p.read_text())['files'].items():
        if digest(name)!=sha:raise RuntimeError(f'An original baseline artifact changed: {name}')


def better(candidate,original):
    from object_motion_quality import nonregressing_objects
    if not nonregressing_objects(candidate,original):return False
    a,b=candidate['single_attempt_tracking'],original['single_attempt_tracking']
    if not a.get('all_frames_finite',True):return False
    if candidate['completion_rate_32_worlds'] < original['completion_rate_32_worlds']-.0625:return False
    new,old=candidate['source_mean_object_iou'],original['source_mean_object_iou']
    previous_objects=original.get('source_metrics',{}).get('objects',{})
    current_objects=candidate.get('source_metrics',{}).get('objects',{})
    if any(name not in current_objects or current_objects[name]['iou']<values['iou']-.025
           for name,values in previous_objects.items()):return False
    if a['passed_all_frames'] and not b['passed_all_frames']:
        return new>=old-.015 and candidate['source_worst_phase_object_iou']>=original['source_worst_phase_object_iou']-.02
    if b['passed_all_frames'] and not a['passed_all_frames']:return False
    if a['passed_all_frames']:
        return (new>=old+.015 or
                candidate['completion_rate_32_worlds']>=original['completion_rate_32_worlds']+.25 and new>=old-.005)
    return (new>=old+.03 and a.get('violation_frame_count',10**9)<=.75*b.get('violation_frame_count',0)
            and candidate['source_worst_phase_object_iou']>=original['source_worst_phase_object_iou']-.02)
