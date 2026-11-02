"""Preserve baseline meshes/19 episodes; replace one complete saved policy rollout.

All exporter digests remain genuine legacy-export provenance. No target poses,
interpolation between attempts, or nonphysical object assistance is introduced.
"""
from pathlib import Path
import hashlib,json,sys
BASE=Path('/home/yu_eli/v2d-track3');sys.path.insert(0,str(BASE))
import batch_submission_stage as stage


def main():
    old=BASE/'runs/evaluation_batch_20261015/submission_staging'
    baseline=json.loads((old/'readiness.json').read_text())
    stage.OUT=BASE/'runs/evaluation_batch_20261015/policy_motion_candidate_20261102'
    stage.OUT.mkdir(exist_ok=True)
    grid,names=stage.public_grid();sources={ep:stage.source(ep) for ep in sorted(grid)}
    for ep,item in sources.items():
        assert item and stage.digest(item['state'])==baseline['episodes'][str(ep)]['identity']['source_state_sha256']
        assert stage.digest(item['checkpoint'])==baseline['episodes'][str(ep)]['identity']['checkpoint_sha256']
    root=BASE/'runs/episode_000019_newton_full_rebuild_20261023'
    selection=root/'final-controller-selection.json';selected=json.loads(selection.read_text())['selected']
    report=json.loads((root/'result-report.json').read_text())
    sources[19]=dict(root=root,recording_root=root,state=root/selected['recorded_states'],
        checkpoint=root/selected['checkpoint'].removeprefix('/episode/'),selection=selection,
        reason='Complete preserved episode 19 policy rollout: improved source-video IoU and independent local motion diagnostics; no frame splicing.',
        quality=report.get('quality_assessment',{}))
    for name in names['19']:
        assert stage.digest(root/f'reconstruction/mesh_priors/{name}.ply')==baseline['episodes']['19']['meshes'][name]['sha256']
    code_hash=stage.code_bundle(sources);episodes={}
    for ep,item in sources.items():
        episodes[str(ep)]=stage.stage_episode(ep,item,grid[ep],names[str(ep)],code_hash,{})
    stage.write_json(stage.OUT/'readiness.json',dict(episodes=episodes,episodes_staged=len(episodes),
        all_20_episode_data_staged=len(episodes)==20,final_submission_package_ready=False,
        note='Legacy policy rollout candidate. Only episode 19 changed; unchanged original meshes. No claim of new official reevaluation or strict completion.',uploaded=False))
    stage.write_json(stage.OUT/'candidate-provenance.json',dict(replaced_episode=19,
        complete_single_attempt=True,source_iou_baseline=.2534628344889116,source_iou_candidate=selected['source_mean_object_iou'],
        no_target_pose_edits=True,source_geometry_unchanged=True,object_assistance_zero=True,
        new_training=False,strict_episode_completion_granted=False))
    print('Prepared full-roster legacy candidate; exactly one complete policy rollout replaced.')


if __name__=='__main__':main()
