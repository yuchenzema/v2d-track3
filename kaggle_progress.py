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
    rows=[]
    for key,name,direction,unit in METRICS:
        item=receipt.get('submissions',{}).get(key,{})
        try:
            score=float(item['public_score'])
            if not math.isfinite(score):score=None
        except (KeyError,TypeError,ValueError):score=None
        rows.append(dict(metric=key,name=name,direction=direction,unit=unit,score=score,
            status=item.get('state','PENDING'),submission_id=item.get('ref'),
            url='https://www.kaggle.com/competitions/'+item['slug'] if item.get('slug') else None))
    return dict(completed=sum(r['status']=='COMPLETE' and r['score'] is not None for r in rows),
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


def latest_comparison(batch):
    batch=Path(batch);baseline=batch/'official_submission_20261102/original20.kaggle-receipt.json'
    candidates=[('Mesh precision',batch/'mesh_precision_candidate_20261102/mesh-precision.kaggle-receipt.json'),
                ('Episode 19 policy',batch/'policy_motion_candidate_20261102/policy-motion.kaggle-receipt.json'),
                ('Combined motion and geometry',batch/'combined_precision_motion_20261102/combined.kaggle-receipt.json')]
    for label,path in reversed(candidates):
        if path.is_file():return dict(compare_receipts(baseline,path),label=label)
    return {}
