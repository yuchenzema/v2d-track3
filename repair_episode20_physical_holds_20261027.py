"""Repair unfrozen hand targets with actual unassisted hold/lift searches."""
import argparse,copy,json,shutil,subprocess
from pathlib import Path
import pyarrow.parquet as pq
from improvement_common import BASE,save,utc,digest
from batch_native import docker

p=argparse.ArgumentParser();p.add_argument('--run',type=Path,required=True);r=p.parse_args().run
c=json.loads((r/'episode-config.json').read_text());assert c['episode']==20
require_clearance=bool(c.get('grasp_search_require_nonpenetrating_initial_pose'))
assert not (r/'coherent-input-checksums.json').exists()
marker=r/'research/physical-holds-verified.json'
if marker.exists():
    prior=json.loads(marker.read_text())
    current_motion=next((r/'human_motion_data/track3_collision_variant/processed').rglob('motion.parquet'))
    bound=not require_clearance or prior.get('reference_parquet_sha256')==digest(current_motion)
    if bound and prior.get('both_objects_static_hold_verified') and (not require_clearance or (set(prior.get('searches',{}))==set(c['objects']) and all(v.get('initial_hand_cross_body_penetration_m',float('inf'))<=1e-6 for v in prior['searches'].values()))):raise SystemExit(0)
    for name,entry in prior.get('independent_retest',{}).get('objects',{}).items():
        applied=r/f'research/physical-hold-{name}-verified.json'
        if (not bound or not entry.get('stable_hold_and_lift')) and applied.exists():
            shutil.move(str(applied),str(r/f'research/physical-hold-{name}-failed-independent-retest.json'))
native=c['native_container'];retarget=native.removesuffix('newton')+'retarget'
reference='track3_collision_variant';motion=next((r/f'human_motion_data/{reference}/processed').rglob('motion.parquet'))
object_columns=['object_body_position','object_body_wxyz']
original=pq.ParquetFile(motion).read(columns=object_columns).to_pylist()
backup=r/'research/motion-before-physical-hold-repair.parquet'
if not backup.exists():shutil.copy2(motion,backup)
fits=json.loads((r/'research/collision-aware-grasps.json').read_text())['grasps']
seeds={g['object']:{**g,'world_inward_direction':[0.,0.,0.],'inward_m':0.} for g in fits}
save(r/'evaluation/grasp-pose-search.json',{'best_candidates':seeds})
shutil.copy2(BASE/'diagnose_refined_grasp_search.py',r/'scripts/diagnose_refined_grasp_search.py')
shutil.copy2(BASE/'rigid_pose_collision_audit.py',r/'scripts/rigid_pose_collision_audit.py')
shutil.copy2(BASE/'grasp_search_protocol.py',r/'scripts/grasp_search_protocol.py')
prior_seed=None
request=json.loads(Path(c['match_request']).read_text()) if c.get('match_request') else {}
seed_run=c.get('grasp_search_prior_run') or request.get('seed_reconstruction_run')
if require_clearance and seed_run:
    prior_root=Path(seed_run).resolve()
    prior_path=prior_root/'research/physical-holds-verified.json'
    if prior_root.is_relative_to(BASE/'runs') and prior_path.exists():
        prior=json.loads(prior_path.read_text())
        if prior.get('both_objects_static_hold_verified') and set(prior.get('searches',{}))==set(c['objects']) and all(digest(prior_root/f'reconstruction/mesh_priors/{obj}.ply')==digest(r/f'reconstruction/mesh_priors/{obj}.ply') for obj in c['objects']):
            prior_seed='research/physical-hold-prior-search-seed.json'
            save(r/prior_seed,{'best_candidates':prior['searches'],'source_run':str(prior_root),
                 'used_only_as_search_seed':True,'current_penetration_gate_still_required':True})

def apply(fit):
    source=next(g for g in fits if g['object']==fit['object'])
    replacement=copy.deepcopy(source);replacement.update(wrist_in_object=fit['wrist_in_object'],finger_joints=fit['finger_joints'],physical_hold_fit=fit)
    save(r/'research/physical-hold-grasp.json',{'grasps':[replacement]})
    text=(BASE/'apply_collision_grasps.py').read_text().replace('human_motion_data/track3_coherent_variant/processed','human_motion_data/track3_collision_variant/processed')
    text=text.replace("R/'research/collision-aware-grasps.json'","R/'research/physical-hold-grasp.json'")
    text=text.replace("R/'research/collision-aware-applied-grasps.json'","R/'research/physical-hold-applied.json'")
    text=text.replace("shutil.copy(p.parent.parent.parent.parent/f'reconstructed_stage/{seq}_support.usda',stage);", "None; # support is already in the same reference\n")
    script=r/'scripts/apply_physical_hold_grasp.py';compile(text,str(script),'exec');script.write_text(text)
    docker(['exec',retarget,'/isaac-sim/python.sh','-u','/episode/scripts/apply_physical_hold_grasp.py'],r/f"logs/apply-physical-hold-{fit['object']}.log")
    assert original==pq.ParquetFile(motion).read(columns=object_columns).to_pylist(),'Object targets changed during robot-grasp repair'

results={}
for name,frame in [('blue_cup',8),('water_pitcher',30)]:
    applied=r/f'research/physical-hold-{name}-verified.json'
    if applied.exists():
        fit=json.loads(applied.read_text())
        if not require_clearance or fit.get('initial_hand_cross_body_penetration_m',float('inf'))<=1e-6:
            results[name]=fit;continue
        shutil.move(str(applied),str(r/f'research/physical-hold-{name}-before-penetration-gate.json'))
    reference_signature=digest(motion)[:8]
    for attempt in range(3):
        label=f'physicalhold20v{4 if require_clearance else 2}-{name.replace("_","-")}-{reference_signature}-{attempt}'
        output=r/f'evaluation/{label}.json'
        env=['V2D_STATIC_GRASP_SOURCE_FRAME='+str(frame),f'V2D_GRASP_SEARCH_GENERATIONS={24 if require_clearance else 12}',
             'V2D_REFINED_GRASP_LABEL='+label,'V2D_GRASP_USE_REFERENCE_SEED=1',
             'V2D_GRASP_SEARCH_OBJECTS='+json.dumps([name]),'V2D_GRASP_CONSISTENT_INITIAL_MOTORS=1',
             'V2D_GRASP_SEARCH_REJECT_INVALID=1','V2D_PROTECT_OTHER_OBJECTS=1',
             'V2D_GRASP_PROBE_REFERENCE='+reference,'V2D_WRIST_MOTOR_PROFILE='+json.dumps(c['wrist_motor_profile'])]
        if attempt:
            env+=['V2D_GRASP_SEED_FIT=research/physical-hold-search-seed.json']
        elif prior_seed:
            env+=['V2D_GRASP_SEED_FIT='+prior_seed]
        if require_clearance:
            env+=['V2D_GRASP_TRANSLATION_BOUND_M=0.09','V2D_GRASP_ROTATION_BOUND_RAD=1.2']
        if not output.exists():
            docker(['exec',native,'env',*env,'python','-u','/episode/scripts/diagnose_refined_grasp_search.py'],r/f'logs/{label}.log')
        report=json.loads(output.read_text());fit=report['best_candidates'][name];fit['object']=name
        save(r/'research/physical-hold-search-seed.json',report)
        if fit['stable_hold_and_lift']:
            apply(fit);save(applied,fit);results[name]=fit;break
    else:raise AssertionError(f'No unassisted stable {name} hold found; do not start another blind100M-step run')

docker(['exec',native,'env','V2D_STATIC_GRASP_SOURCE_FRAME=30','V2D_GRASP_PROBE_REFERENCE='+reference,
        'V2D_WRIST_MOTOR_PROFILE='+json.dumps(c['wrist_motor_profile']),
        'python','-u','/episode/scripts/diagnose_static_grasp.py'],r/'logs/physical-hold-independent-retest.log')
retest=json.loads((r/'evaluation/static-grasp-hold-lift-source-30-v2-kp250-kd20.json').read_text())
passed=set(retest['objects'])==set(c['objects']) and all(v.get('stable_hold_and_lift',False) for v in retest['objects'].values())
report={'updated_utc':utc(),'both_objects_static_hold_verified':passed,'diagnostic_only':True,
            'policy_inference':False,'cold_start_policy_pickup_verified':False,'source_video_match_verified':False,
            'object_trajectories_unchanged':True,'searches':results,'independent_retest':retest}
if not passed:save(marker,report)
assert passed,'Applied grips failed the independent static retest; policy training remains blocked'
# Reapplying a grip also blends into the withdrawal interval. Restore the
# source-timed finger opening so the new grasp cannot delay the release.
release_marker=r/'research/rebuilt-release-targets.json'
if release_marker.exists():
    shutil.copy2(release_marker,r/'research/release-targets-before-physical-grip-repair.json');release_marker.unlink()
subprocess.run([str(r/'.venv-video/bin/python'),str(BASE/'rebuild_release_targets_20261022.py'),'--run',str(r)],check=True)
docker(['exec',retarget,'env','V2D_TRANSITION_REFERENCE='+reference,'V2D_REBUILD_ONLY=1',
        'V2D_CONTACT_APPLY_OUTPUT=research/physical-grip-release-fk.json','/isaac-sim/python.sh','-u','/episode/scripts/apply_collision_transitions.py'],r/'logs/physical-grip-release-fk.log')
subprocess.run([str(r/'.venv-video/bin/python'),str(BASE/'trim_release_contact_targets.py'),'--run',str(r)],check=True)
assert original==pq.ParquetFile(motion).read(columns=object_columns).to_pylist()
report['reference_parquet_sha256']=digest(motion)
save(marker,report)
print('Both static holds pass independently. Cold-start learned pickup and source motion remain unverified.',flush=True)
