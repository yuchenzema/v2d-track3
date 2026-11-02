"""Report object agreement within visually annotated human task events."""
import argparse,json
from pathlib import Path

def assess(root,row):
 root=Path(root);ep=json.loads((root/'episode-config.json').read_text())['episode']
 review=Path('/home/yu_eli/v2d-track3/runs')/f'episode_{ep:06d}_newton_grasp_curriculum_20261020/research/source-human-video-recheck-20261020.json'
 if not review.exists():return None
 label=Path(row['recorded_states']).stem.removesuffix('-states');p=root/'evaluation'/f'{label}-source-fidelity.json';fidelity=json.loads(p.read_text());events=[]
 for event in json.loads(review.read_text())['events']:
  frames=[f for f in fidelity['frames'] if event['start_source_frame']<=f['frame']<=event['end_source_frame']]
  assert len(frames)==event['end_source_frame']-event['start_source_frame']+1
  objects={}
  for name in frames[0]['objects']:
   values=[f['objects'][name]['iou'] for f in frames]
   objects[name]={'mean_iou':sum(values)/len(values),'minimum_iou':min(values),'frames_below_0_5_iou':sum(v<.5 for v in values),'source_frames':len(values)}
  events.append({**event,'objects':objects})
 result={'episode':ep,'method':__doc__,'recorded_states':row['recorded_states'],'source_event_review':str(review),'events':events,
  'interpretation':'Diagnostic source-pixel agreement by human event; exact motion matching and official evaluation remain separate.'}
 return result
if __name__=='__main__':
 p=argparse.ArgumentParser();p.add_argument('--run',type=Path,required=True);p.add_argument('--review',type=Path,required=True);p.add_argument('--output',type=Path,required=True);a=p.parse_args();row=json.loads(a.review.read_text());row=row.get('candidate',row);result=assess(a.run,row);assert result is not None;a.output.write_text(json.dumps(result,indent=2));print(json.dumps(result,indent=2))
