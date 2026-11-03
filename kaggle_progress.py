"""Display submitted Kaggle scores independently from local episode completion."""
import json
import math
from decimal import Decimal
from pathlib import Path

METRICS=[('add_auc','AUC','higher',''),('spider_sr','SP-SR','higher',''),
         ('maniptrans_sr','MP-SR','higher',''),('rpe_cm','RPE','lower','cm'),
         ('mppe_cm','MPPE','lower','cm'),('cd_o_cm','CD-O','lower','cm')]


def receipt_scores(path):
    try:receipt=json.loads(Path(path).read_text())
    except (OSError,ValueError):return {}
    if not receipt:return {}
    live={}
    for parent in Path(path).resolve().parents:
        cache=parent/'kaggle-live-status.json'
        if cache.is_file():
            try:live=json.loads(cache.read_text())
            except (OSError,ValueError):pass
            break
    rows=[]
    for key,name,direction,unit in METRICS:
        item=receipt.get('submissions',{}).get(key,{})
        try:
            score=float(item['public_score'])
            if not math.isfinite(score):score=None
        except (KeyError,TypeError,ValueError):score=None
        status=item.get('state','PENDING');error=None;verified=False
        current=live.get('metrics',{}).get(key,{})
        matched=next((s for s in current.get('submissions',[]) if str(s.get('ref'))==str(item.get('ref'))),None)
        if matched:
            status=matched.get('state','UNKNOWN');error=matched.get('error');verified=True
            try:score=float(matched['public_score']) if status=='COMPLETE' else None
            except (KeyError,ValueError,TypeError):score=None
            if score is not None and not math.isfinite(score):score=None
        rows.append(dict(metric=key,name=name,direction=direction,unit=unit,score=score,
            status=status,submission_id=item.get('ref'),error=error,live_verified=verified,
            url='https://www.kaggle.com/competitions/'+item['slug'] if item.get('slug') else None))
    return dict(completed=sum(r['status']=='COMPLETE' and r['score'] is not None for r in rows),
                invalidated=sum('invalidated' in (r['error'] or '') for r in rows),
                live_checked_utc=live.get('checked_utc'),
                total=6,metrics=rows,receipt=str(path),counts_as_strict_episode_completion=False)


def compare_receipts(baseline,candidate):
    before=receipt_scores(baseline);after=receipt_scores(candidate)
    if not before or not after:return {}
    rows=[]
    for old,new in zip(before['metrics'],after['metrics']):
        row=dict(new,baseline=old['score'],verdict='pending',change=None)
        if old['status']=='COMPLETE' and new['status']=='COMPLETE' and old['score'] is not None and new['score'] is not None:
            change=Decimal(str(new['score']))-Decimal(str(old['score']))
            gain=change if new['direction']=='higher' else -change
            row.update(change=float(change),verdict='improved' if gain>0 else 'regressed' if gain<0 else 'unchanged')
        rows.append(row)
    counts={name:sum(r['verdict']==name for r in rows) for name in ['improved','unchanged','regressed','pending']}
    return dict(metrics=rows,completed=6-counts['pending'],total=6,**counts,
                all_six_strictly_improved=counts['improved']==6,
                non_regressing=counts['pending']==0 and counts['regressed']==0,
                receipt=str(candidate),counts_as_strict_episode_completion=False)


def display_scores(batch,receipt):
    """Keep one complete cohort visible while a different cohort is scoring."""
    latest=receipt_scores(receipt)
    if latest.get('completed')==6:return latest
    paths=list((Path(batch)/'own_geometry_submissions').glob('*/kaggle-receipt.json'))
    def produced(path):
        provenance=path.parent/'cohort-provenance.json'
        return (provenance if provenance.exists() else path).stat().st_mtime
    for path in sorted(paths,key=produced,reverse=True):
        previous=receipt_scores(path)
        if previous.get('completed')==6:
            return dict(previous,pending_candidate=latest)
    return latest


def latest_comparison(batch):
    batch=Path(batch);baseline=batch/'official_submission_20261102/original20.kaggle-receipt.json'
    status=batch/'focused_episodes_20261028/own-submission-status.json'
    try:own=json.loads(status.read_text())
    except (OSError,ValueError):own={}
    if own.get('receipt') and Path(own['receipt']).is_file():
        shown=display_scores(batch,own['receipt']);candidate=shown.get('receipt') or own['receipt']
        try:policy=json.loads((batch/'focused_episodes_20261028/quality-improvement-policy.json').read_text())
        except (OSError,ValueError):policy={}
        if policy.get('baseline_receipt') and Path(policy['baseline_receipt']).is_file():
            return dict(compare_receipts(policy['baseline_receipt'],candidate),label='Last fully scored quality cohort versus valid own-geometry baseline',pending_candidate=shown.get('pending_candidate'))
        return dict(compare_receipts(baseline,own['receipt']),label='Own reconstruction cohort; invalid baseline is not comparable')
    candidates=[('Mesh precision',batch/'mesh_precision_candidate_20261102/mesh-precision.kaggle-receipt.json'),
                ('Episode 19 policy',batch/'policy_motion_candidate_20261102/policy-motion.kaggle-receipt.json'),
                ('Combined motion and geometry',batch/'combined_precision_motion_20261102/combined.kaggle-receipt.json')]
    for label,path in reversed(candidates):
        if path.is_file():return dict(compare_receipts(baseline,path),label=label)
    return {}


def current_scores(batch):
    batch=Path(batch)
    try:status=json.loads((batch/'focused_episodes_20261028/own-submission-status.json').read_text())
    except (OSError,ValueError):status={}
    return display_scores(batch,status.get('receipt') or batch/'official_submission_20261102/original20.kaggle-receipt.json')
