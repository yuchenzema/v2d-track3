"""Pack complete genuine cohorts while changing declared episode selections only."""
import argparse,hashlib,json,shutil,subprocess
from pathlib import Path
from improvement_common import BASE,BATCH,save,utc
from own_submission_queue import eligible,PY,KIT


def run(parent_cohort,replacement_cohort,episodes,metrics):
    old=json.loads(Path(parent_cohort).read_text());new=json.loads(Path(replacement_cohort).read_text())
    assert old['public_scans_used'] is False and new['public_scans_used'] is False
    rows=dict(old['episodes']);assert len(rows)==20 and set(rows)==set(new['episodes'])
    for ep in episodes:rows[str(ep)]=new['episodes'][str(ep)]
    for row in rows.values():assert eligible(row['run'],row.get('result_override'))==row,'Immutable episode artifacts changed'
    identity=dict(episodes=rows,metrics=metrics)
    tag=hashlib.sha256(json.dumps(identity,sort_keys=True).encode()).hexdigest()[:16]
    destination=BATCH/'quality_ablations_20261103'/tag;destination.mkdir(parents=True,exist_ok=True)
    receipt=destination/'kaggle-receipt.json'
    if receipt.exists():return receipt
    assert not subprocess.check_output(['git','status','--porcelain','--untracked-files=no'],cwd=BASE,text=True).strip(),'Commit producing code first'
    commit=subprocess.check_output(['git','rev-parse','HEAD'],cwd=BASE,text=True).strip()
    for ep,row in rows.items():
        results=destination/'results';results.mkdir(exist_ok=True)
        shutil.copy2(row['trajectory'],results/f'episode_{int(ep):06d}.parquet');shutil.copy2(row['official_report'],results/f'episode_{int(ep):06d}.json')
        meshes=destination/f'meshes/episode_{int(ep):06d}';meshes.mkdir(parents=True,exist_ok=True)
        for name,path in row['meshes'].items():shutil.copy2(path,meshes/f'{name}.glb')
    save(destination/'cohort-provenance.json',dict(episodes=rows,producing_code_commit=commit,public_scans_used=False,
        parent_cohort=str(parent_cohort),replacement_cohort=str(replacement_cohort),replaced_episodes=episodes,
        source_clock_preserved=True,whole_episodes_only=True,metrics=metrics,updated_utc=utc()))
    package=destination/'track3_submission.parquet';description='Full evaluation-only own-geometry cohort; declared whole-episode selection comparison'
    commands=[['tools/pack_submission.py','--track','track_3','--results',str(results),'--meshes',str(destination/'meshes'),
        '--code-commit',f'https://github.com/yuchenzema/v2d-track3/commit/{commit}','-o',str(package)],
        ['tools/preflight.py','--track','track_3','--results',str(results)],
        ['submit/v2d_submit.py',str(package),'--track','track_3','--only',*metrics,'-m',description,'--dry-run']]
    for index,args in enumerate(commands):
        with (destination/f'preflight-{index}.log').open('w') as log:subprocess.run([str(PY),str(KIT/args[0]),*args[1:]],cwd=KIT,stdout=log,stderr=subprocess.STDOUT,check=True)
    for row in rows.values():assert eligible(row['run'],row.get('result_override'))==row
    subprocess.run(['git','push','origin','HEAD:main'],cwd=BASE,capture_output=True,text=True,check=True)
    with (destination/'upload.log').open('a') as log:
        subprocess.run([str(PY),str(KIT/'submit/v2d_submit.py'),str(package),'--track','track_3','--only',*metrics,
            '-m',description,'--receipt',str(receipt)],cwd=KIT,stdout=log,stderr=subprocess.STDOUT,check=True)
    return receipt


if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('--parent',type=Path,required=True);parser.add_argument('--replacement',type=Path,required=True)
    parser.add_argument('--episodes',type=int,nargs='+',required=True);parser.add_argument('--only',nargs='+',required=True)
    args=parser.parse_args();print(run(args.parent,args.replacement,args.episodes,args.only),flush=True)
