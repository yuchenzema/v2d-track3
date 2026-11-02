"""Compare bounded integration/controller alternatives before the first video export."""
import json,shutil,traceback
from batch_worker import context
from batch_review import review

def optimize(ep):
 r,c=context(ep);marker=r/'research/integration-controller-selection.json'
 if (r/'result-report.json').exists() or marker.exists():return
 path=r/'final-controller-selection.json';selection=json.loads(path.read_text());baseline=selection['selected'];substeps=baseline.get('physics_substeps',c.get('physics_substeps',5));gain=baseline.get('wrist_reference_velocity_gain',0.);candidates=[];errors=[]
 if substeps<20 and not (baseline['single_attempt_tracking']['passed_all_frames'] and baseline['completion_rate_32_worlds']>=.96875 and baseline['source_worst_phase_object_iou']>=.7):
  trials=[(10,gain),(20,gain)] if substeps==5 else [(20,gain),(20,.5 if gain!=.5 else 0.)]
  for steps,feedforward in trials:
   try:
    override=baseline['checkpoint'] if '/research/' in baseline['checkpoint'] else None;row=review(ep,baseline['environment_steps'],feedforward,baseline.get('inference_reference',c.get('training_reference','track3_coherent_variant')),steps,checkpoint_override=override);candidates.append(row);print(ep,'integration candidate',20*steps,feedforward,'completion',row['completion_rate_32_worlds'],'source',row['source_mean_object_iou'],flush=True)
   except Exception as error:errors.append({'substeps':steps,'gain':feedforward,'error':str(error),'traceback':traceback.format_exc()})
 eligible=[baseline]
 for candidate in candidates:
  if not candidate['single_attempt_tracking']['passed_all_frames'] or candidate['completion_rate_32_worlds']<max(.25,baseline['completion_rate_32_worlds']-.125):continue
  score=candidate['source_agreement_score'];old=baseline['source_agreement_score']
  if (not baseline['single_attempt_tracking']['passed_all_frames'] and score>=old-.03) or (score>old+.005 and candidate['source_mean_object_iou']>=baseline['source_mean_object_iou']-.01) or (candidate['completion_rate_32_worlds']>=baseline['completion_rate_32_worlds']+.25 and score>=old-.015):eligible.append(candidate)
 passing=[row for row in eligible if row['single_attempt_tracking']['passed_all_frames']]
 selected=max(passing or eligible,key=lambda x:x['source_agreement_score'])
 if selected is not baseline:
  shutil.copy2(path,r/'research/controller-selection-before-integration-search.json');selection['selected']=selected;selection['integration_controller_candidates']=candidates;tmp=path.with_suffix('.integration.tmp');tmp.write_text(json.dumps(selection,indent=2));tmp.replace(path)
 decision={'method':__doc__,'baseline':baseline,'candidates':candidates,'errors':errors,'selected':selected,'changed_from_baseline':selected is not baseline,'maximum_tested_integration_hz':400,'source_timestamps_and_frozen_training_inputs_preserved':True,'virtual_object_guidance_at_evaluation':0.,'candidate_criteria':'Require one complete finite unassisted attempt and no material completion regression; improve source agreement, or gain substantial completion without material source regression. Preserve earlier selection otherwise.'};marker.write_text(json.dumps(decision,indent=2));print(ep,'integration/controller optimization complete; changed',selected is not baseline,flush=True)
