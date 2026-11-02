"""Bounded robot wrist tracking review after the 10/16 learners completed.

Keep the GUI's accepted policy unless the existing per-object improvement gate
passes. Objects remain free physical bodies and the original source clock stays
unchanged. The larger cohort is independently checked before publication.
"""
from pathlib import Path
from batch_review import review
from improvement_common import BASE, BATCH, better, save, utc
from video_catalog import catalog
from publish_contact_controller_candidate import publish


def main():
    results = []
    output = BATCH/'improvements_20261020_contacts/completed-feedforward-review.json'
    for ep, variant, step, gains in [
        (10, 'stable_contact_20261020', 370000384, (.25, .75)),
        (16, 'stationary_hold_20261020', 380000256, (.25, .5)),
    ]:
        root = BASE/'runs'/f'episode_{ep:06d}_newton_{variant}'
        checkpoint = root/f'research/checkpoint-preview-{step}/policy_{step}.safetensors'
        for gain in gains:
            current = catalog(ep, BASE/'runs'/f'episode_{ep:06d}_newton_batch_20261015', True)['best']
            row = review(ep, step, gain=gain, root_override=root,
                         container_override='v2d-stable-episode10-newton' if ep==10 else 'v2d-stationary-episode16-newton',
                         checkpoint_override='/episode/'+str(checkpoint.relative_to(root)),
                         substeps_override=20)
            qualifies = better(row, current['selected'])
            label = f'completed-{step}-gain-{str(gain).replace(".", "p")}'
            path = root/'research'/f'{label}-review.json'
            save(path, row)
            result = {'episode':ep, 'gain':gain, 'review':str(path),
                      'source_mean_object_iou':row['source_mean_object_iou'],
                      'completion_rate_32_worlds':row['completion_rate_32_worlds'],
                      'better_than_accepted':qualifies, 'published':False}
            print(result, flush=True)
            if qualifies:
                published = publish(ep,path,label,root_override=root)
                result['published']=True
                result['completion_rate_128_worlds']=published['quality_assessment']['completion_rate_against_video_estimate']
                result['accepted_version']=catalog(ep,BASE/'runs'/f'episode_{ep:06d}_newton_batch_20261015')['latest']['version']
            results.append(result)
            save(output,{'phase':'reviewing','updated_utc':utc(),'results':results})
    save(output,{'phase':'completed','updated_utc':utc(),'results':results,
                 'existing_acceptance_gates_unchanged':True,'object_assistance':0.})


if __name__=='__main__':
    main()
