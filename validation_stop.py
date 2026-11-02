"""Stop additional training only after two strong, independently recorded checkpoint reviews."""
from pathlib import Path
import hashlib,json,time,sys

class TrainingValidated(Exception):
 pass

def maybe_validation_stop(root,step,start):
 request=Path(root)/'research/validation-stop-request.json'
 if not request.exists():return
 payload=json.loads(request.read_text());selected=payload['checkpoint_step'];folder=Path(root)/'flash_chord_coherent_actor_transfer'
 assert selected-start>=30_000_000
 for kind in ['policy','state']:
  path=folder/f'{kind}_{selected}.safetensors'
  if not path.exists():return
 path=folder/f'policy_{selected}.safetensors';assert hashlib.sha256(path.read_bytes()).hexdigest()==payload['checkpoint_sha256']
 payload.update(actual_environment_steps=int(step),actual_additional_environment_steps=int(step)-start,maximum_budget_replaced_by_validated_early_stop=True)
 (Path(root)/'research/training-early-stop.json').write_text(json.dumps(payload,indent=2))
 raise TrainingValidated('Two consecutive full-motion validation checkpoints satisfy the stopping criteria')

def scan(base,include_improvements=False):
 roots=list((base/'runs').glob('episode_*_newton_batch_20261015'))
 if include_improvements:
  roots+=list((base/'runs').glob('episode_*_newton_grasp_curriculum_20261020'))
  roots+=list((base/'runs').glob('episode_*_newton_backward_grasp_20261020'))
 for r in roots:
  folder=r/'flash_chord_coherent_actor_transfer';request=r/'research/validation-stop-request.json'
  if (folder/'TRAINING_COMPLETED').exists() or request.exists():continue
  script=r/'scripts/train_coherent_actor_transfer.py'
  if not script.exists() or 'VALIDATED_STOP_PROTOCOL=2' not in script.read_text():continue
  p=r/'checkpoint-reviews.json'
  if not p.exists():continue
  try:rows=sorted(json.loads(p.read_text()),key=lambda x:x['additional_environment_steps'])
  except (OSError,ValueError):continue
  if len(rows)<2:continue
  recent=rows[-2:];source=json.loads((r/'evaluation/reference-source-fidelity.json').read_text())['aggregate']['objects'];baseline=sum(v['iou'] for v in source.values())/len(source)
  original_source_fit=baseline;contact_reference=None;terminal_start=None
  episode_config=json.loads((r/'episode-config.json').read_text())
  if episode_config.get('training_curriculum')=='contact_finetune':
   decision=json.loads((r/'research/physical-grasp-reference-selection.json').read_text())
   path=(r/decision['physical_reference_source_report']).resolve();assert path.is_relative_to(r.resolve())
   contact_reference=json.loads(path.read_text())
   assert decision['approved_for_independent_training'] and contact_reference['reference']==episode_config['training_reference']
   assert contact_reference['source_frames']==episode_config['frames']
   baseline=sum(v['mean_iou'] for v in contact_reference['aggregate'].values())/len(contact_reference['aggregate'])
   terminal_start=json.loads((r/'research/source-human-video-recheck-20261020.json').read_text())['events'][-1]['start_source_frame']
  def terminal_contact_match(x):
   if contact_reference is None:return True
   label=Path(x['recorded_states']).stem.removesuffix('-states')
   try:frames=json.loads((r/'evaluation'/f'{label}-source-fidelity.json').read_text())['frames']
   except (OSError,ValueError,KeyError):return False
   for name in contact_reference['aggregate']:
    expected=[f['objects'][name]['iou'] for f in contact_reference['frames'] if f['source_frame']>=terminal_start]
    actual=[f['objects'][name]['iou'] for f in frames if f['frame']>=terminal_start]
    if len(actual)!=len(expected) or not actual:return False
    if sum(actual)/len(actual)<max(.55,sum(expected)/len(expected)-.08):return False
   return True
  def good(x):
   tracking=x['single_attempt_tracking']
   return x['additional_environment_steps']>=30_000_000 and x['completion_rate_32_worlds']==1. and tracking['passed_all_frames'] and x['source_mean_object_iou']>=baseline-.06 and x['source_worst_phase_object_iou']>=.55 and max(v['max_position_error_m'] for v in tracking['per_body'].values())<=.05 and terminal_contact_match(x)
  if not all(good(x) for x in recent):continue
  if not r.name.endswith('_newton_batch_20261015'):
   from improvement_common import baseline as original_selection
   from object_motion_quality import nonregressing_objects
   episode=json.loads((r/'episode-config.json').read_text())['episode']
   previous=original_selection(episode)
   if not all(nonregressing_objects(x,previous) for x in recent):continue
  x=max(recent,key=lambda x:x['source_agreement_score']);step=x['environment_steps'];cp=folder/f'policy_{step}.safetensors'
  if not cp.exists() or not (folder/f'state_{step}.safetensors').exists():continue
  provenance=r/'coherent-training-provenance.json'
  budget=json.loads(provenance.read_text()).get('additional_training_step_budget',100003840) if provenance.exists() else 100003840
  payload={'criterion':__doc__,'checkpoint_step':step,'checkpoint_sha256':hashlib.sha256(cp.read_bytes()).hexdigest(),'reviewed_steps':[v['environment_steps'] for v in recent],'completion_rate_each_32_world_cohort':1.,'both_recorded_attempts_pass_every_native_frame':True,'source_baseline_mean_iou':baseline,'selected_mean_iou':x['source_mean_object_iou'],'minimum_worst_phase_iou':.55,'maximum_object_position_error_m':.05,'minimum_new_training_steps':30_000_000,'original_maximum_budget':budget,'utc':time.strftime('%Y-%m-%dT%H:%M:%SZ',time.gmtime())}
  if contact_reference is not None:
   payload.update(source_benchmark_type='Reviewed physically feasible contact reference; original source pixels remain the scoring target.',
    original_unconstrained_source_fit_mean_iou=original_source_fit,
    reviewed_reference_source_report=decision['physical_reference_source_report'],
    final_human_event_checked_for_every_object=True,final_event_start_source_frame=terminal_start,
    final_event_mean_iou_minimum=.55,final_event_maximum_iou_deficit_from_reviewed_reference=.08)
  temporary=request.with_suffix('.tmp');temporary.write_text(json.dumps(payload,indent=2));temporary.replace(request);print(r.name,'validated training stop requested at',step,flush=True)

if __name__=='__main__':
 base=Path('/home/yu_eli/v2d-track3')
 while True:
  scan(base,include_improvements='--include-improvements' in sys.argv)
  if '--watch' not in sys.argv:break
  time.sleep(30)
