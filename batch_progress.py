"""Publish factual status from completed artifacts and actual training counters."""
from pathlib import Path
import json,time,sys,os
from batch_artifacts import complete
BASE=Path('/home/yu_eli/v2d-track3');B=BASE/'runs/evaluation_batch_20261015'
def update():
 rows=[]
 for p in sorted((BASE/'runs').glob('episode_*_newton_batch_20261015/episode-config.json')):
  c=json.loads(p.read_text());r=p.parent;status=json.loads((r/'pipeline-status.json').read_text()) if (r/'pipeline-status.json').exists() else {};review=json.loads((r/'checkpoint-reviews.json').read_text()) if (r/'checkpoint-reviews.json').exists() else [];metric=r/'flash_chord_coherent_actor_transfer/metrics.jsonl';steps=0
  if metric.exists():
   lines=metric.read_text().splitlines()
   for line in reversed(lines):
    try:steps=json.loads(line)['additional_environment_steps'];break
    except (ValueError,KeyError):pass
  report=json.loads((r/'result-report.json').read_text()) if complete(r) else None
  row={'episode':c['episode'],'source_frames':c['frames'],'original_frame_review_complete':c['all_frames_visually_reviewed'],'reconstruction':status.get('reconstruction',{}).get('status','pending'),'reconstruction_visual_reviewed':c.get('reconstruction_visual_reviewed',False),'retarget':status.get('retarget',{}).get('status','pending'),'additional_training_steps':steps,'training_completed':(r/'flash_chord_coherent_actor_transfer/TRAINING_COMPLETED').exists(),'reviewed_checkpoints':len(review),'rollout_export_completed':report is not None,'run':str(r)}
  row['pipeline_status']=status.get('pipeline',{}).get('status','pending')
  if row['pipeline_status']=='failed':row['pipeline_error']=status['pipeline'].get('traceback','')
  if review:row['best_32_world_completion_so_far']=max(v['completion_rate_32_worlds'] for v in review)
  if report:row['quality_assessment']=report['quality_assessment']
  rows.append(row)
 payload={'updated_utc':time.strftime('%Y-%m-%dT%H:%M:%SZ',time.gmtime()),'episodes_requested':18,'excluded_episodes':[1,5],'original_frames_reviewed':sum(c['source_frames'] for c in rows if c['original_frame_review_complete']),'reconstructions_completed':sum(c['reconstruction']=='completed' for c in rows),'policies_training_completed':sum(c['training_completed'] for c in rows),'rollout_exports_completed':sum(c['rollout_export_completed'] for c in rows),'free_disk_gb':round(os.statvfs(BASE).f_bavail*os.statvfs(BASE).f_frsize/1e9,2),'episodes':rows}
 payload['failed_episode_ids']=[c['episode'] for c in rows if c['pipeline_status']=='failed' and not c['rollout_export_completed']]
 temp=B/f'batch-status.{os.getpid()}.tmp';temp.write_text(json.dumps(payload,indent=2));temp.replace(B/'batch-status.json');return payload
if __name__=='__main__':
 while True:
  d=update();print(json.dumps({k:v for k,v in d.items() if k!='episodes'}),flush=True)
  if '--watch' not in sys.argv or d['rollout_exports_completed']==18:break
  time.sleep(30)
