"""Review every completed contact-finetuning checkpoint against human task events."""
import json,time,fcntl,traceback
from pathlib import Path
from assess_human_events import assess
from improvement_common import BASE,BATCH,save,utc,root,baseline
OUT=BATCH/'improvements_20261020_contacts'
def scan():
 results={}
 for ep in [10,16]:
  r=BASE/'runs'/f'episode_{ep:06d}_newton_grasp_curriculum_20261020';p=r/'checkpoint-reviews.json'
  if not p.exists():results[ep]={'status':'waiting_for_reviewed_checkpoints'};continue
  old=assess(root(ep),baseline(ep));rows=[]
  for checkpoint in json.loads(p.read_text()):
   report=assess(r,checkpoint)
   if report is None:continue
   for event,previous in zip(report['events'],old['events']):
    assert event['name']==previous['name']
    for name,value in event['objects'].items():value['mean_iou_change_from_preserved_baseline']=value['mean_iou']-previous['objects'][name]['mean_iou']
   rows.append({'environment_steps':checkpoint['environment_steps'],'additional_environment_steps':checkpoint['additional_environment_steps'],**report})
  save(r/'research/human-event-checkpoint-reviews.json',{'updated_utc':utc(),'checkpoints':rows,'reference':'research/source-human-video-recheck-20261020.json'})
  results[ep]={'status':'reviewed','checkpoints':len(rows),'latest_additional_environment_steps':rows[-1]['additional_environment_steps'] if rows else None}
 save(OUT/'human-events-status.json',{'updated_utc':utc(),'episodes':results,'diagnostic_only':True,'policy_selection_gates_unchanged':True})
if __name__=='__main__':
 with (OUT/'human-events-watcher.lock').open('a') as lock:
  fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
  while True:
   try:scan()
   except (OSError,ValueError,KeyError):print('Human event review retry',traceback.format_exc(),flush=True)
   time.sleep(20)
