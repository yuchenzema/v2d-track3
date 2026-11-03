"""Compare exact submitted receipts, excluding invalidated or pending scores."""
import json
from pathlib import Path
from improvement_common import save,utc


def compare(out,baseline_receipt):
    out=Path(out);baseline=json.loads(Path(baseline_receipt).read_text());status=json.loads((out/'own-submission-status.json').read_text());receipt=Path(status['receipt']);candidate=json.loads(receipt.read_text());rows={}
    higher={'add_auc','spider_sr','maniptrans_sr'}
    for metric,old in baseline['submissions'].items():
        new=candidate.get('submissions',{}).get(metric,{})
        valid=new.get('state')=='COMPLETE' and old.get('state')=='COMPLETE' and not new.get('error') and not old.get('error')
        a=float(old['public_score']) if valid else None;b=float(new['public_score']) if valid else None
        rows[metric]=dict(baseline=a,candidate=b,valid_scored_comparison=valid,improved=valid and ((b>a) if metric in higher else (b<a)),submission_id=new.get('ref'))
    result=dict(updated_utc=utc(),baseline_receipt=str(baseline_receipt),candidate_receipt=str(receipt),metrics=rows,
        all_six_improved=len(rows)==6 and all(r['improved'] for r in rows.values()))
    save(out/'quality-kaggle-comparison.json',result);return result
