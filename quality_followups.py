"""Admit measured follow-ups after every evaluation episode has had its turn."""
import json
from pathlib import Path
from improvement_common import save,utc


def read(path):
    try:return json.loads(Path(path).read_text())
    except (OSError,ValueError):return {}


def improving(records):
    records=sorted(records,key=lambda r:r['actual_training_steps'])
    if len(records)<2:return False
    old,new=records[-2:]
    return (new['source_video_loss']<old['source_video_loss']*.95
        and new.get('rollout_motion',{}).get('dropped_frames',0)<=old.get('rollout_motion',{}).get('dropped_frames',0))


def apply_repair_requests(out,progress):
    """Apply explicit measured repairs once, at a serial controller boundary."""
    from own_geometry import validate
    path=Path(out)/'quality-repair-requests.json';requests=read(path)
    for key,request in requests.get('episodes',{}).items():
        if request.get('applied_utc'):continue
        row=progress.get('episodes',{}).get(key,{})
        if row.get('status')!='completed' or row.get('run')!=request.get('source_run'):continue
        root=Path(row['run']);cfg=read(root/'episode-config.json')
        if not cfg.get('own_geometry_required') or cfg.get('actor_transfer'):continue
        validate(root)
        progress.setdefault('followup_history',[]).append(dict(episode=int(key),kind='measured_contact_reference_repair',previous=dict(row),basis=request['basis'],updated_utc=utc()))
        progress['episodes'][key]=dict(status='contact_refit_pending',prior_run=str(root),contact_recipe_revision=3)
        request['applied_utc']=utc()
    if requests:save(path,requests);save(Path(out)/'quality-training-status.json',progress)


def admit(out,focus,progress,policy):
    from own_geometry import validate
    from training_continuation import extend
    admitted=[];history=progress.setdefault('followup_history',[])
    maximum=int(policy.get('quality_followup_maximum_steps',5120000))
    for ep in focus:
        row=progress['episodes'][str(ep)];root=Path(row['run']);cfg=read(root/'episode-config.json')
        result=read(root/'research/fast-submission-result.json')
        if result.get('local_reference_metrics',{}).get('chord_sr')==1.:continue
        if not cfg.get('own_geometry_required') or cfg.get('actor_transfer'):continue
        validate(root)
        source=Path(cfg.get('fast_source_run',''))
        contact=read(root/'research/quality-contact-reference.json')
        if (source.name.endswith('_quality_20261103_v1')
                or contact.get('optimizer_revision')!='supporting_contact_wrench_v3'):
            # A fresh source algorithm needs a fresh experiment, not edits to
            # geometry or contacts behind a learned checkpoint.
            history.append(dict(episode=ep,kind='fresh_source_and_contact_reference',previous=dict(row),updated_utc=utc()))
            progress['episodes'][str(ep)]=dict(status='source_refit_pending',prior_run=str(root),source_algorithm_revision=2)
            admitted.append(ep);continue
        if sum(h.get('kind')=='measured_optimizer_continuation' for h in history)>=int(policy.get('maximum_measured_followups',4)):continue
        records=[read(p) for p in (root/'research/quality-checkpoint-validation').glob('*.json')]
        records=[r for r in records if r.get('validation_revision')==2 and r.get('actual_training_steps')]
        if not records or max(r['actual_training_steps'] for r in records)>=maximum or not improving(records):continue
        previous_limit=cfg['own_training_maximum_steps'];cfg['own_training_maximum_steps']=maximum
        save(root/'episode-config.json',cfg)
        try:target=extend(root,increment=min(1280000,maximum-max(r['actual_training_steps'] for r in records)))
        except Exception:
            cfg['own_training_maximum_steps']=previous_limit;save(root/'episode-config.json',cfg);raise
        history.append(dict(episode=ep,kind='measured_optimizer_continuation',previous_limit=previous_limit,
            new_limit=maximum,target_steps=target,basis='At least 5% full-source validation improvement with no increase in dropped frames',updated_utc=utc()))
        row.update(status='continuation_pending');admitted.append(ep)
    save(Path(out)/'quality-training-status.json',progress)
    return admitted
