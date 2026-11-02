"""Publish a validated full 400 Hz attempt while retaining the complete prior result."""
from pathlib import Path
import json,shutil,os
from batch_export import export
R=Path('/home/yu_eli/v2d-track3/runs/episode_000003_newton_batch_20261015');candidate=json.loads((R/'evaluation/higher-rate-policy-trial-20.json').read_text());new=json.loads((R/'evaluation/final-400hz-metrics.json').read_text());old=json.loads((R/'evaluation/final-metrics.json').read_text());assert candidate['single_attempt_tracking']['passed_all_frames'] and candidate['source_mean_object_iou']>.555735 and new['metrics']['chord_sr']>old['metrics']['chord_sr'] and not new['non_finite_worlds'];B=R/'research/rollout_100hz_baseline';assert not B.exists();B.mkdir();(B/'videos').mkdir();(B/'evaluation').mkdir()
shutil.copytree(R/'deliverables',B/'deliverables',symlinks=True)
for name in ['reconstruction','flash_chord_coherent_actor_transfer']:(B/name).symlink_to(os.path.relpath(R/name,B),target_is_directory=True)
for name in ['result-report.json','final-controller-selection.json']:shutil.copy2(R/name,B/name)
selection=json.loads((R/'final-controller-selection.json').read_text());oldstate=R/selection['selected']['recorded_states'];shutil.copy2(oldstate,B/'evaluation'/oldstate.name)
for name in ['final-metrics.json','final-trajectories.parquet']:(R/'evaluation'/name).rename(B/'evaluation'/name)
for name in ['original_vs_policy','original_vs_reconstruction','policy_rollout_source_timed','trained_policy_source_camera','trained_policy_rollout','reconstruction_reference']:
 for suffix in ['.mp4','.protocol.json']:
  p=R/'videos'/(name+suffix)
  if p.exists():p.rename(B/'videos'/p.name)
# Kinematic reconstruction has identical geometry and camera after this change.
for name in ['reconstruction_reference.mp4','reconstruction_reference.protocol.json']:
 p=B/'videos'/name
 if p.exists():shutil.copy2(p,R/'videos'/name)
shutil.copy2(R/'evaluation/final-400hz-metrics.json',R/'evaluation/final-metrics.json');shutil.copy2(R/'evaluation/final-400hz-trajectories.parquet',R/'evaluation/final-trajectories.parquet');selection['controller_candidates'].append(candidate);selection['selected']=candidate;selection['physics_validation']={'previous_integration_hz':100,'selected_integration_hz':400,'previous_128_world_completion':old['metrics']['chord_sr'],'selected_128_world_completion':new['metrics']['chord_sr'],'previous_nonfinite_world_count':len(old['non_finite_worlds']),'selected_nonfinite_world_count':len(new['non_finite_worlds']),'previous_result_preserved':str(B.relative_to(R))};(R/'final-controller-selection.json').write_text(json.dumps(selection,indent=2));export(3)
