"""Expose a complete, audited candidate as a diagnostic; never replace Latest or Best."""
import copy,hashlib,json,subprocess
from pathlib import Path
from improvement_common import BASE,BATCH,save,utc,verify_preserved,baseline
from object_motion_quality import attach
from render_improved_controller import render
R=BASE/'runs/episode_000010_newton_grasp_curriculum_20261020'
row=copy.deepcopy(json.loads((R/'research/half-new-left-preserved-right-policy-review.json').read_text()))
states=R/row['recorded_states'];metrics=R/row['recorded_states'].replace('-states.npz','-metrics.json');native=json.loads(metrics.read_text())
assert '/track3_collision_variant/' in native['parquet']
original=BASE/'runs/episode_000010_newton_batch_20261015'
candidate_parquet=next((R/'human_motion_data/track3_collision_variant').rglob('motion.parquet'))
baseline_parquet=next((original/'human_motion_data/track3_collision_variant').rglob('motion.parquet'))
assert hashlib.sha256(candidate_parquet.read_bytes()).hexdigest()==hashlib.sha256(baseline_parquet.read_bytes()).hexdigest()
output=R/'research/compatibility-reference-half-ensemble-tracking.json'
subprocess.run([str(R/'.venv-video/bin/python'),str(R/'scripts/assess_single_attempt.py'),'--states',str(states),'--reference','track3_collision_variant','--output',str(output)],check=True)
row.update(inference_reference='track3_collision_variant',single_attempt_tracking=json.loads(output.read_text()),
 reference_protocol_mismatch=False,publication_eligible=False,
 reference_protocol_audit_resolved={'actual_reference':native['parquet'],'source':'Preserved checkpoint metadata, recorded command and native evaluation report agree. Reference parquet matches the unchanged baseline copy.','physics_states_modified':False})
row.pop('object_motion_quality',None);row=attach(R,row)
save(R/'research/compatibility-reference-half-ensemble-review.json',row)
dst=R/'research/review-only-half-ensemble';rendered=render(10,row,dst,root_override=R,container_override='v2d-grasp-episode10-newton')
verify_preserved(10)
result={'episode':10,'status':'completed','release_type':'diagnostic','updated_utc':utc(),'evaluation_run':str(R),
 'label':'Candidate · held outer cup still needs work','selected':row,'baseline':baseline(10),'rendered':rendered,
 'measured_improvement':False,'strict_improvement_checks_passed':False,
 'native_32_world_evaluation':native,'quality_assessment':{'all_internal_frame_checks_passed':False,
  'completion_rate_against_video_estimate':native['metrics']['chord_sr'],'validation_cohort_worlds':32,
  'remaining_problem':'Outer-cup held-phase overlap regresses despite better full-video overlap and stable final nesting.'},
 'full_human_motion_match_verified':False,'eligible_for_latest':False,'training_completed':False}
save(BATCH/'improvements_20261020_contacts/episode_000010-review-only-half-ensemble-result.json',result)
print('Review-only complete videos available; Latest and Best selections remain unchanged.',flush=True)
