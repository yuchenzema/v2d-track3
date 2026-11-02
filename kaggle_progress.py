"""Display submitted Kaggle scores independently from local episode completion."""
import json
import math
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
