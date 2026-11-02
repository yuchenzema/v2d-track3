"""Publish an updating local video gallery with factual per-episode results."""
from pathlib import Path
import html,json,time,sys
from batch_artifacts import complete
BASE=Path('/home/yu_eli/v2d-track3');B=BASE/'runs/evaluation_batch_20261015'
def update():
 status=json.loads((B/'batch-status.json').read_text());cards=[]
 for row in status['episodes']:
  r=Path(row['run']);ep=row['episode'];relative='../'+r.name;report=r/'result-report.json';finished=complete(r);result=json.loads(report.read_text()) if finished else None
  state='Videos ready' if finished else ('Failed: needs recovery' if row.get('pipeline_status')=='failed' else 'Selecting / rendering' if row['training_completed'] else 'Training' if row['additional_training_steps'] else 'Reconstruction review' if not row['reconstruction_visual_reviewed'] else 'Queued')
  metrics=''
  if result:
   q=result['quality_assessment'];tracking=result['selected_attempt_reference_tracking'];overlap=result['selected_source_fidelity']['objects'];mean=sum(v['iou'] for v in overlap.values())/len(overlap)
   metrics=f'<p>Final 128-world completion: <b>{q["completion_rate_against_video_estimate"]:.1%}</b> · Source overlap: {mean:.1%}<br>Selected attempt passes every native tracking frame: <b>{"yes" if tracking["passed_all_frames"] else "no"}</b><br>All internal frame checks passed: <b>{"yes" if q.get("all_internal_frame_checks_passed",q["full_human_video_match_verified"]) else "no"}</b></p>'
  body=f'<article><h2>Episode {ep:06d} <small>{state}</small></h2><p>{row["source_frames"]} human-video frames · {row["additional_training_steps"]/1e6:.1f} million new training steps</p>{metrics}'
  if finished:
   body+=f'<video controls preload="none" src="{relative}/videos/original_vs_policy.mp4"></video><p>Human video and learned-policy physics, shown at source timing.</p><p><a href="{relative}/videos/original_vs_policy.mp4">Comparison</a> · <a href="{relative}/videos/policy_rollout_source_timed.mp4">Rollout</a> · <a href="{relative}/videos/original_vs_reconstruction.mp4">Reconstruction comparison</a> · <a href="{relative}/videos/trained_policy_rollout.mp4">3-D view</a> · <a href="{relative}/result-report.json">Detailed results</a> · <a href="{relative}/deliverables/manifest.json">Artifact checksums</a></p>'
  body+='</article>';cards.append(body)
 refresh='<meta http-equiv="refresh" content="60">' if status['rollout_exports_completed']<18 else ''
 text=f'''<!doctype html><html lang="en"><meta charset="utf-8">{refresh}<meta name="viewport" content="width=device-width,initial-scale=1"><title>Track 3 evaluation videos</title><style>body{{font:16px system-ui,sans-serif;background:#10141c;color:#e9edf4;max-width:1200px;margin:auto;padding:32px}}h1{{font-size:28px}}h2{{font-size:20px}}small{{font-size:13px;color:#99d7b9;margin-left:12px}}a{{color:#9bd2ff}}p{{line-height:1.6;color:#c8d0dd}}main{{display:grid;grid-template-columns:repeat(auto-fit,minmax(360px,1fr));gap:20px}}article{{padding:20px;border:1px solid #354052;border-radius:12px;min-width:0}}video{{width:100%;background:#070a0e;aspect-ratio:8/3}}.note{{font-size:14px}}@media(max-width:430px){{body{{padding:16px}}main{{display:block}}article{{margin-bottom:16px}}}}</style><h1>Track 3 evaluation videos</h1><p><b>{status['rollout_exports_completed']} of 18 episodes exported</b> · {status['reconstructions_completed']} reconstructed · {status['policies_training_completed']} training runs finished<br>Updated {html.escape(status['updated_utc'])}</p><p class="note">Each policy video shows one continuous physical attempt, with no object guidance, failure resets, or frame splicing. Reconstruction videos show estimated reference poses. Scores use the inferred video motion and internal checks; they are not official challenge scores or a guarantee of pixel-perfect human imitation. Comparisons follow source timing; native 3-D playback uses the training recipe's half-speed physics. Earlier checkpoints are retained when they perform better.</p><main>{''.join(cards)}</main></html>'''
 temporary=B/'videos.html.tmp';temporary.write_text(text);temporary.replace(B/'videos.html');return status['rollout_exports_completed']
if __name__=='__main__':
 while True:
  try:update()
  except (FileNotFoundError,json.JSONDecodeError):pass
  if '--watch' not in sys.argv:break
  time.sleep(30)
