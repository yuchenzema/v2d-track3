"""Wait for the existing serial replay, then upload the preserved 20 policies.

This command never starts an episode worker or uploads a draft export.
"""
import argparse
import fcntl
import hashlib
import json
from pathlib import Path
import subprocess
import time

BASE=Path(__file__).resolve().parent
QUEUE=BASE/'runs/evaluation_batch_20261015/focused_episodes_20261028'
EXPECTED={1,3,5,6,8,10,14,15,16,19,20,25,28,30,32,33,35,36,37,38}
PYTHON=BASE/'.venv-submission/bin/python'
UPLOADER=BASE/'submission_kit_20261102/submit/v2d_submit.py'


def read(path):
    try:return json.loads(Path(path).read_text())
    except (OSError,ValueError):return {}


def sha(path):
    h=hashlib.sha256()
    with Path(path).open('rb') as f:
        for b in iter(lambda:f.read(1<<20),b''):h.update(b)
    return h.hexdigest()


def ready_package(request,status):
    if status.get('status')=='blocked':
        raise RuntimeError('官方重評／打包遇到錯誤，沒有上傳：'+str(status.get('error'))+
                           '\n請查看 '+str(status.get('failure_log')))
    if status.get('status')!='complete':return None
    assert {int(ep) for ep in status.get('episodes',{})}==EXPECTED,'Official replay does not cover all 20 episodes'
    assert status.get('code_commit')==request.get('code_commit'),'Replay commit differs from the requested producing code'
    package=status.get('package',{})
    assert package.get('official_packer_used') is True and package.get('dry_run_passed') is True,'Official packing/preflight incomplete'
    path=Path(package['file']).resolve()
    assert path==Path(request['output']).resolve()/'track3_submission.parquet','Unexpected submission path'
    assert sha(path)==package['sha256'],'Submission file changed after verification'
    return path


def poll_command(receipt):
    return [str(PYTHON),str(UPLOADER),'--track','track_3','--poll-receipt',str(receipt)]


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--status',action='store_true',help='Read progress only; never upload')
    parser.add_argument('--poll',action='store_true',help='Poll an existing receipt without uploading')
    args=parser.parse_args()
    request=read(QUEUE/'submission-replay-request.json')
    assert request.get('output'),'Missing existing official replay request'
    destination=Path(request['output']);receipt=destination/'original20.kaggle-receipt.json'
    if args.status:
        s=read(QUEUE/'submission-replay-status.json');state=read(QUEUE/'status.json')
        print(json.dumps(dict(replay_status=s.get('status','waiting_for_serial_controller'),
            officially_replayed=len(s.get('episodes',{})),required=20,
            current_episode=state.get('current_episode'),phase=state.get('phase'),
            package=s.get('package'),receipt_exists=receipt.exists()),ensure_ascii=False,indent=2));return 0
    destination.mkdir(parents=True,exist_ok=True)
    with (destination/'original20-upload.lock').open('a') as lock:
        try:fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
        except BlockingIOError:raise RuntimeError('已有一個原版 20 集提交／監看程序，請沿用原本的 terminal。')
        if receipt.exists():
            previous=read(receipt)
            if not any(entry.get('ref') for entry in previous.get('submissions',{}).values()):
                raise RuntimeError('已有提交 receipt，但沒有可確認的 submission ID；請先檢查 Kaggle，避免重複提交。\n'+str(receipt))
            print('沿用已存在的 receipt，只查詢分數，不重新提交。',flush=True)
            return subprocess.call(poll_command(receipt),cwd=BASE)
        if args.poll:raise RuntimeError('尚無 receipt；沒有執行任何上傳。')
        policy=read(QUEUE/'own-geometry-policy.json')
        if policy.get('enabled'):
            raise RuntimeError('舊版提交使用公開掃描衍生網格，已禁止重新上傳。請沿用 evaluation 自建幾何流程；既有 receipt 仍可唯讀查詢。')
        print('提交原本 20 集。等待既有 controller 的官方重評與打包；背景改善持續。',flush=True)
        print('進度每 10 分鐘顯示；Ctrl+C 結束本提交監看，背景 controller 繼續。',flush=True)
        last_report=0
        while True:
            request=read(QUEUE/'submission-replay-request.json');status=read(QUEUE/'submission-replay-status.json')
            assert request.get('enabled'),'Existing official replay request is disabled'
            if status and status.get('request_sha256')!=sha(QUEUE/'submission-replay-request.json'):
                status={}
            package=ready_package(request,status)
            if package:break
            if time.monotonic()-last_report>=600:
                subprocess.run([str(BASE/'run_evaluation20.sh'),'status'],cwd=BASE,check=True)
                print(f'提交用官方重評 {len(status.get("episodes",{}))}/20；尚未上傳。',flush=True)
                last_report=time.monotonic()
            time.sleep(15)
        print('官方重評及格式預檢完成；再次驗證後提交到六個 Track 3 metric。',flush=True)
        common=[str(PYTHON),str(UPLOADER),str(package),'--track','track_3',
                '-m','Original 20 preserved policies; official evaluation replay']
        subprocess.run([*common,'--dry-run'],cwd=BASE,check=True)
        assert sha(package)==status['package']['sha256'],'Submission changed during preflight'
        return subprocess.call([*common,'--wait','--receipt',str(receipt)],cwd=BASE)


if __name__=='__main__':
    try:raise SystemExit(main())
    except KeyboardInterrupt:
        print('\n提交監看已結束；背景 controller 繼續。若上傳曾開始，重新執行會沿用 receipt。')
        raise SystemExit(130)
    except (AssertionError,RuntimeError,OSError,subprocess.CalledProcessError) as error:
        print(str(error));raise SystemExit(2)
