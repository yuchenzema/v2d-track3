"""Score every source frame of a newly trained rollout and refresh GUI selections."""
import argparse
import json
import os
from pathlib import Path
from improvement_common import BATCH, save, utc
from video_catalog import run_release
import thorough_episode_recheck as human

p=argparse.ArgumentParser();p.add_argument('--run',type=Path,required=True);r=p.parse_args().run
c=json.loads((r/'episode-config.json').read_text());out=Path(c.get('match_output',BATCH/'improvements_20261023_rebuild'))
result=json.loads((out/f"episode_{c['episode']:06d}-fresh-result.json").read_text())
human.OUT=out/'human_review';human.OUT.mkdir(exist_ok=True);human.__file__=__file__
previous=[]
roster=c.get('audit_roster',[19,20,30,33])
for episode in roster:
    path=human.OUT/f'episode_{episode:06d}/report.json'
    if episode!=c['episode'] and path.exists():previous.append(json.loads(path.read_text()))
def progress(completed,ep=None,frame=0,total=0,status='running',**extra):
    save(human.OUT/'status.json',{'updated_utc':utc(),'pid':os.getpid(),'process_script':Path(__file__).name,
         'status':status,'episodes_completed':len(previous),'total_episodes':len(roster),'current_episode':ep,
         'current_frame':frame,'current_total_frames':total,
         'source_frames_completed':sum(item['source_frames'] for item in previous if item['episode']!=ep)+frame})
    if ep is not None:
        save(BATCH/f'improvements_20261021/activities/episode_{ep:06d}-human-review-20261023.json',
             {'episode':ep,'pid':os.getpid(),'process_script':Path(__file__).name,'updated_utc':utc(),
              'status':status,'phase':f'New policy vs original human: {frame} / {total} source frames',
              'completed_cases':frame,'total_cases':total,'activity_kind':'human_video_frame_review'})
human.report_status=progress
release=run_release(c['episode'],r,'experiment',result)
summary=human.recheck({'episode':c['episode'],'source_frames':c['frames'],'objects':c['objects'],'videos':{'latest':release}},previous)
summary.update(independent_fresh_training_completed=True,new_policy_training_performed=True,
               reconstruction_rerun_from_original_video=True,audit_itself_performed_training=False,matches_every_human_frame_verified=False)
save(human.OUT/f"episode_{c['episode']:06d}/report.json",summary)
previous.append(summary)
progress(previous,ep=c['episode'],frame=c['frames'],total=c['frames'],status='completed')
from policy_recommendations import snapshot
snapshot(finalize=False)
from batch_submission_stage import update
update()
