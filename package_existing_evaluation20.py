"""Snapshot existing evaluation artifacts without granting quality or submission approval."""
from pathlib import Path
import ast
import hashlib
import json
import shutil
import sys
import tarfile

BASE=Path(__file__).resolve().parent
STAGE=BASE/'runs/evaluation_batch_20261015/submission_staging'
REVIEW=BASE/'submission_review_20261102'
DEST=Path('/home/yu_eli/v2d_evaluation20_delivery_20261102')
KIT=REVIEW/'v2d_submission_kit'
sys.path.insert(0,str(KIT))
import numpy as np
import pandas as pd
import pyarrow.parquet as pq
from v2dlb.track3_pose import pack_object_meshes,validate_packed_meshes


def sha(p):
    h=hashlib.sha256()
    with Path(p).open('rb') as stream:
        for chunk in iter(lambda:stream.read(1<<20),b''):h.update(chunk)
    return h.hexdigest()


def save(p,d):
    p.parent.mkdir(parents=True,exist_ok=True);p.write_text(json.dumps(d,indent=2,ensure_ascii=False))


def copy(source,target):
    source=Path(source)
    if source.is_dir():
        for p in sorted(source.rglob('*')):
            if p.is_file() and '__pycache__' not in p.parts:copy(p,target/p.relative_to(source))
        return
    target.parent.mkdir(parents=True,exist_ok=True)
    before=sha(source)
    if not target.exists() or sha(target)!=before:shutil.copy2(source,target)
    assert sha(target)==before and sha(source)==before,f'Concurrent change: {source}'


def main():
    DEST.mkdir(exist_ok=True)
    readiness=json.loads((STAGE/'readiness.json').read_text())
    assert readiness['episodes_staged']==20
    roster=json.loads((KIT/'data/track_3_evaluation_objects.json').read_text())['episodes']
    assert set(readiness['episodes'])==set(roster)
    copy(KIT,DEST/'kit');copy(REVIEW/'kit-file-inventory.json',DEST/'reports/kit-file-inventory.json')
    review=[]
    for f in sorted(KIT.rglob('*')):
        if not f.is_file() or '__pycache__' in f.parts:continue
        row={'path':str(f.relative_to(KIT)),'sha256':sha(f),'bytes':f.stat().st_size}
        if f.suffix=='.py':
            tree=ast.parse(f.read_text());row['definitions']=[n.name for n in ast.walk(tree) if isinstance(n,(ast.FunctionDef,ast.ClassDef))]
            row['module_contract']=ast.get_docstring(tree)
        elif f.suffix=='.parquet':
            pf=pq.ParquetFile(f);row.update(rows=pf.metadata.num_rows,schema=str(pf.schema_arrow))
        elif f.suffix=='.csv':
            table=pd.read_csv(f);row.update(rows=len(table),columns=list(table.columns))
        else:row['text']=f.read_text()
        review.append(row)
    save(DEST/'reports/complete-kit-review.json',review)
    copy(STAGE,DEST/'staged')
    entries={};frames=[];checkpoints={}
    for ep,item in readiness['episodes'].items():
        e=int(ep);prefix=DEST/f'episodes/episode_{e:06d}'
        checkpoint=Path(item['checkpoint']);recording=Path(item['recording'])
        copy(checkpoint,prefix/'policy.safetensors');copy(recording,prefix/'recorded-policy-states.npz')
        root=recording.parent.parent
        for name in ['episode-config.json','final-controller-selection.json','result-report.json','source_reference_entrypoint.py']:
            if (root/name).exists():copy(root/name,prefix/name)
        for name in ['scripts','human_motion_data','reconstruction/mesh_priors','deliverables']:
            if (root/name).exists():copy(root/name,prefix/name)
        entries[ep]={'checkpoint':str(checkpoint),'checkpoint_sha256':sha(checkpoint),
                     'recording':str(recording),'recording_sha256':sha(recording),
                     'quality':item.get('quality'),'identity':item.get('identity'),
                     'caveat':'Legacy physical rollout export, not a new organizer-evaluator output.'}
        checkpoints[ep]=f'episodes/episode_{e:06d}/policy.safetensors'
        t=pd.read_parquet(STAGE/f'results/episode_{e:06d}.parquet')
        t['row_id']=[f'track_3/e{e:06d}/w{w:03d}/f{f:06d}/o{s}' for w,f,s in zip(t.world_index,t.frame_index,t.object_slot)]
        for col in ['checkpoint_sha256','eval_script_sha256','eval_code_sha256']:
            t[col]=item['identity'][col] if col!='eval_code_sha256' else readiness.get('unused','')
        # Preserve original exporter provenance directly from the file metadata.
        metadata=pq.read_metadata(STAGE/f'results/episode_{e:06d}.parquet').metadata
        t['eval_code_sha256']=metadata[b'eval_code_sha256'].decode()
        frames.append(t)
        print(f'Snapshotted episode {e}',flush=True)
    table=pd.concat(frames,ignore_index=True)
    sample=pd.read_parquet(KIT/'data/track_3_sample_submission.parquet')
    assert table.row_id.is_unique and set(table.row_id)==set(sample.row_id)
    table=table.set_index('row_id').loc[sample.row_id].reset_index()
    pose=['pos_x','pos_y','pos_z','quat_x','quat_y','quat_z','quat_w']
    assert np.isfinite(table[pose].to_numpy()).all()
    assert np.allclose(np.linalg.norm(table[pose[3:]].to_numpy(),axis=1),1,atol=1e-4)
    table=pack_object_meshes(table,DEST/'staged/meshes',roster)
    meshes=validate_packed_meshes(table)
    assert len(meshes)==sum(map(len,roster.values()))
    table.to_parquet(DEST/'evaluation20_results_with_meshes.DRAFT.parquet',index=False)
    save(DEST/'checkpoints.json',checkpoints);save(DEST/'reports/selected-artifacts.json',entries)
    pilot=BASE/'runs/episode_000038_newton_human_match_20261023_g003'
    for folder in ['deliverables','evaluation/official','research']:
        if (pilot/folder).exists():copy(pilot/folder,DEST/'experimental_episode38'/folder)
    for f in sorted(BASE.glob('*.py')):copy(f,DEST/'code_snapshot'/f.name)
    for pattern in ['*.sh','*.txt']:
        for f in sorted(BASE.glob(pattern)):copy(f,DEST/'code_snapshot'/f.name)
    queue=BASE/'runs/evaluation_batch_20261015/focused_episodes_20261028'
    for n in ['source-cad-status.json','experimental-training-status.json','experimental-training-policy.json','status.json']:
        copy(queue/n,DEST/'reports'/n)
    status=dict(artifact_snapshot_complete=True,episodes=20,pose_rows=len(table),object_meshes=len(meshes),
                strict_completion_count=0,uploaded=False,official_submission_ready=False,container_verified=False,
                blockers=['Producing code has no supplied GitHub commit URL / clean producing checkout.',
                          'Legacy staged evaluator hashes identify an export adapter, not the required organizer evaluator. Re-evaluate all 20 checkpoints with the pinned evaluator before formal submission.',
                          'Kaggle entry/selected submission IDs and receipts have not been checked.',
                          'Reproducible offline container verification has not passed.'],
                format_validation='Exact full public row roster, finite poses, unit XYZW quaternions, all object mesh payloads decoded by the provided kit.',
                quality_note='All known motion/contact failures remain. Draft archive does not certify successful imitation or eligibility.')
    save(DEST/'DELIVERY_STATUS.json',status)
    (DEST/'READ_ME_FIRST.txt').write_text('''Evaluation 20 成果快照 — 2026-11-02

這是現有成果交付包，不是已通過比賽提交或容器驗證的成品。
包含舊版 20 集模型、真實物理錄影狀態、來源資料與腳本、影片、mesh、逐集品質與 provenance。
experimental_episode38 是另外的新流程試驗，不取代選定的舊版 episode 38。
嚴格完成仍為 0/20；不能把 CAD 已審核視為成功。

evaluation20_results_with_meshes.DRAFT.parquet 已驗證完整 row roster 與所有 mesh，
但刻意不填造假的 code_commit_url，也不把舊 exporter 的 hash 冒充官方 evaluator。
因此此 DRAFT 檔不能直接當成正式 Kaggle submission。

正式交付流程（詳見 kit/tracks/track_3/README.md）：
1. 將產生結果的程式納入可供主辦方檢視的 GitHub commit；保存當時環境。
2. 在相容 FlashCHORD runtime 用 kit/examples/run_policy_evaluation.py 評估 20 個模型。
   checkpoints.json 是本包內相對路徑；從本包根目錄執行。
3. 用原封未改的 kit/tools/pack_submission.py 打包正式軌跡和 staged/meshes，傳入真實 --code-commit。
4. preflight、Kaggle 規則／帳戶／額度與 dry-run 檢查後，提交到 Track 3 六個 metric，保留 receipt。
5. 依 kit/CONTAINER_SUBMISSION.md 製作離線可重現容器；verify 與 reproduce 全通過後，
   docker save 產生容器檔，隨技術報告交付可下載連結、SHA256 和 image digest。

reports/complete-kit-review.json 包含完整 kit 64 個檔案的雜湊、所有文字文件、程式契約和資料 schema。
verify_snapshot.py 可在解壓後驗證每份檔案，所有路徑都包含於壓縮檔，不依賴外部 symlink。
DELIVERY_STATUS.json 列出目前尚缺的正式交付條件。
''')
    verifier='''from pathlib import Path
import hashlib,json
p=Path(__file__).resolve().parent
files=json.loads((p/'SHA256_MANIFEST.json').read_text())
for name,expected in files.items():
 h=hashlib.sha256()
 with (p/name).open('rb') as f:
  for b in iter(lambda:f.read(1048576),b''):h.update(b)
 assert h.hexdigest()==expected,name
print('Verified',len(files),'files')
'''
    (DEST/'verify_snapshot.py').write_text(verifier)
    files={str(p.relative_to(DEST)):sha(p) for p in sorted(DEST.rglob('*')) if p.is_file() and p.name!='SHA256_MANIFEST.json'}
    assert not any(p.is_symlink() for p in DEST.rglob('*'))
    save(DEST/'SHA256_MANIFEST.json',files)
    archive=DEST.with_suffix('.tar.gz');temporary=archive.with_suffix('.tmp')
    print('Archiving',len(files),'files',flush=True)
    with tarfile.open(temporary,'w:gz',compresslevel=1) as tar:tar.add(DEST,arcname=DEST.name)
    temporary.replace(archive)
    archive.with_suffix(archive.suffix+'.sha256').write_text(sha(archive)+'  '+archive.name+'\n')
    print(json.dumps(dict(archive=str(archive),bytes=archive.stat().st_size,sha256=sha(archive),**status),ensure_ascii=False),flush=True)


if __name__=='__main__':main()
