"""Preserve the next complete learner checkpoint, then adopt verified initial support."""
import json,os,signal,subprocess,time,shutil
from pathlib import Path
from improvement_common import BASE,BATCH,save,utc,digest,better,baseline
from batch_native import _docker
from prepare_contact_precision_run import prepare

OLD=BASE/'runs/episode_000030_newton_observed_brush_20261021'
NEW=BASE/'runs/episode_000030_newton_stable_brush_20261021'
OUT=BATCH/'improvements_20261021'


def main():
    assert json.loads((OLD/'research/stable-brush-pickup-reference.json').read_text())['approved_for_independent_training']
    cfg=json.loads((OLD/'episode-config.json').read_text());start=cfg['training_initial_counter'];target=None
    while target is None:
        reviews=json.loads((OLD/'checkpoint-reviews.json').read_text()) if (OLD/'checkpoint-reviews.json').exists() else []
        for row in reviews:
            step=row['environment_steps'];path=OLD/f'flash_chord_coherent_actor_transfer/state_{step}.safetensors'
            if step-start>=4_900_000 and path.exists() and time.time()-path.stat().st_mtime>5:target=step;break
        save(OUT/'stable-brush-handoff.json',{'phase':'waiting_for_saved_and_reviewed_checkpoint','updated_utc':utc(),'minimum_additional_steps':4_900_000,'preserved_training_run':str(OLD)})
        if target is None:time.sleep(15)
    row=next(v for v in reviews if v['environment_steps']==target)
    if better(row,baseline(30)):
        # Let the existing coordinator finish its native validation and video
        # export before stopping its learner. Never orphan a GPU rendering job.
        release=OUT/f'episode_000030-checkpoint-{target}-result.json';deadline=time.monotonic()+1800
        while not release.exists():
            assert time.monotonic()<deadline,'Verified preview did not finish before the checkpoint handoff'
            time.sleep(5)
    last=json.loads((OLD/'flash_chord_coherent_actor_transfer/metrics.jsonl').read_text().splitlines()[-1])
    code="""import os,signal
from pathlib import Path
found=[]
for p in Path('/proc').glob('[0-9]*/cmdline'):
 try:a=p.read_bytes().split(b'\\0')
 except OSError:continue
 if len(a)>2 and a[1]==b'-u' and a[2]==b'/episode/scripts/train_coherent_actor_transfer.py':found.append(int(p.parent.name))
assert len(found)==1,found
os.kill(found[0],signal.SIGTERM)
"""
    _docker(['exec',cfg['native_container'],'python','-c',code])
    previous_worker=json.loads((OUT/'episode_000030-worker.json').read_text())['pid']
    for _ in range(45):
        path=Path(f'/proc/{previous_worker}/stat')
        if not path.exists() or path.read_text().split()[2]=='Z':break
        time.sleep(1)
    else:raise RuntimeError('Previous coordinator still owns results')
    remaining=cfg['contact_training_additional_budget']-(target-start)
    prepare(30,OLD,target,'track3_stable_brush_pickup_candidate','stable_brush_20261021',remaining,[24,72],full_state=True)
    # Keep the successful training objective and optimizer state; only the
    # independently checked initial support reference changes at this boundary.
    shutil.copy2(OLD/'research/contact-finetune-adjustments.json',NEW/'research/contact-finetune-adjustments.json')
    for folder,name in [('research','stable-brush-pickup-reference.json'),('research','initial-brush-support-search.json'),
                        ('evaluation','stable-brush-initial-rest.json'),('evaluation','stable-brush-source-fidelity.json')]:
        shutil.copy2(OLD/folder/name,NEW/folder/name)
    selection=json.loads((NEW/'research/physical-grasp-reference-selection.json').read_text());selection.update(reference='track3_stable_brush_pickup_candidate',
        physical_reference_source_report='evaluation/stable-brush-source-fidelity.json',initial_support_evidence='evaluation/stable-brush-initial-rest.json')
    save(NEW/'research/physical-grasp-reference-selection.json',selection)
    save(OLD/'research/superseded-training.json',{'updated_utc':utc(),'superseded_by_run':str(NEW),'saved_checkpoint_environment_steps':target,
         'state_sha256':digest(OLD/f'flash_chord_coherent_actor_transfer/state_{target}.safetensors'),
         'last_observed_environment_steps':last['environment_steps'],'unsaved_steps_not_carried_forward':last['environment_steps']-target,
         'reason':'Adopt the independently stable, source-scored initial brush support pose. Existing actor/critic/optimizer/normalizer are carried forward.'})
    registry=json.loads((OUT/'active-runs.json').read_text()) if (OUT/'active-runs.json').exists() else {'32':'episode_000032_newton_contact_precision_full_20261021'}
    registry['30']=NEW.name;save(OUT/'active-runs.json',registry)
    with (OUT/'episode_000030-stable-worker.log').open('a') as log:
        worker=subprocess.Popen([str(BASE/'.venv/bin/python'),'-u',str(BASE/'run_contact_precision_episode.py'),'30','stable_brush_20261021'],cwd=BASE,stdout=log,stderr=subprocess.STDOUT,start_new_session=True)
    save(OUT/'episode_000030-worker.json',{'pid':worker.pid,'started_utc':utc(),'episode':30,'variant':'stable_brush_20261021'})
    save(OUT/'stable-brush-handoff.json',{'phase':'stable_reference_continuation_started','updated_utc':utc(),'saved_environment_steps':target,
         'remaining_additional_training_steps':remaining,'full_learner_state_preserved':True,'run':str(NEW)})
    print('Stable brush continuation started from the complete reviewed checkpoint:',target,flush=True)


if __name__=='__main__':main()
