"""Durable learning, whole-source checkpoint review and verified video publication."""
import argparse,json,threading,traceback
from improvement_common import BASE,BATCH,save,utc,better
from video_catalog import catalog
from fresh_improvement_run import run
from publish_checkpoint_preview import publish


def main(ep,variant):
    root=BASE/'runs'/f'episode_{ep:06d}_newton_{variant}'
    out=BATCH/'improvements_20261021';out.mkdir(exist_ok=True)
    stopped=threading.Event();errors={};published=set()
    def previews():
        while not stopped.is_set():
            path=root/'checkpoint-reviews.json'
            if path.exists():
                try:
                    best=catalog(ep,BASE/'runs'/f'episode_{ep:06d}_newton_batch_20261015',True)['best']['selected']
                    for row in json.loads(path.read_text()):
                        step=row['environment_steps']
                        if step in published or errors.get(step,0)>=2 or not better(row,best):continue
                        try:
                            publish(ep,step,variant,result_directory=out);published.add(step)
                        except Exception:
                            errors[step]=errors.get(step,0)+1;print('Preview review:',traceback.format_exc(),flush=True)
                except (OSError,ValueError,KeyError):print(traceback.format_exc(),flush=True)
            stopped.wait(20)
    worker=threading.Thread(target=previews,daemon=True);worker.start()
    try:
        result=run(ep,variant)
        save(out/f'episode_{ep:06d}-completion.json',result)
    finally:
        stopped.set();worker.join(timeout=60)


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('episode',type=int);p.add_argument('variant');a=p.parse_args();main(a.episode,a.variant)
