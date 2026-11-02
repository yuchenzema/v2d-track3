"""Render an explicitly unaccepted spoon checkpoint for visual diagnosis."""
import copy,json,shutil,os
from improvement_common import BASE,BATCH,save,utc,baseline,verify_preserved
from render_improved_controller import render
R=BASE/'runs/episode_000016_newton_stationary_hold_20261020';STEP=int(os.environ.get('V2D_REVIEW_STEP','370000384'))
row=copy.deepcopy(next(a for a in json.loads((R/'checkpoint-reviews.json').read_text()) if a['environment_steps']==STEP))
millions=round(row['additional_environment_steps']/1e6)
dst=R/f'research/review-only-stationary-{millions}M';dst.mkdir(exist_ok=True)
original=R/row['checkpoint'].removeprefix('/episode/');archive=dst/original.name
if not archive.exists():shutil.copy2(original,archive)
row['checkpoint']='/episode/'+str(archive.relative_to(R))
native=json.loads((R/row['recorded_states'].replace('-states.npz','-metrics.json')).read_text())
rendered=render(16,row,dst,root_override=R,container_override='v2d-stationary-episode16-newton')
verify_preserved(16)
save(BATCH/f'improvements_20261020_contacts/episode_000016-review-only-stationary-{millions}M-result.json',{
 'episode':16,'status':'completed','release_type':'diagnostic','updated_utc':utc(),'evaluation_run':str(R),
 'label':f'Candidate · {millions}M steps; spoon release still needs work','selected':row,'baseline':baseline(16),'rendered':rendered,
 'measured_improvement':False,'strict_improvement_checks_passed':False,'native_32_world_evaluation':native,
 'quality_assessment':{'all_internal_frame_checks_passed':False,'completion_rate_against_video_estimate':native['metrics']['chord_sr'],
 'validation_cohort_worlds':32,'remaining_problem':'Pickup and held-motion overlap recovered, but the spoon still falls after release.'},
 'full_human_motion_match_verified':False,'eligible_for_latest':False,'training_completed':False})
print('Complete spoon candidate videos available for review; not accepted as an improvement.',flush=True)
