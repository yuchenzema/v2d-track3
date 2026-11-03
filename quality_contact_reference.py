"""Fit new own-geometry contact targets before freezing a quality experiment."""
import json,shutil
from pathlib import Path
import pyarrow.parquet as pq
from improvement_common import BASE,save,utc,digest
from batch_native import _docker,docker


def fit(root,native,retarget):
    root=Path(root);cfg=json.loads((root/'episode-config.json').read_text());marker=root/'research/quality-contact-reference.json'
    if marker.exists():return
    assert not (root/'coherent-input-checksums.json').exists(),'Keep trained references immutable'
    reference=root/f"human_motion_data/track3_coherent_variant/processed/sequence_id={cfg['sequence']}/robot_name=sharpa_wave/motion.parquet"
    before=pq.ParquetFile(reference).read();source_sha=digest(reference)
    for name in ['native_pinch_contacts.py','grasp_physics_checks.py','rigid_pose_collision_audit.py','run_geometry_cpu.py']:
        shutil.copy2(BASE/name,root/'scripts'/name)
    text=(BASE/'optimize_native_grasps.py').read_text()
    text=text.replace('f"track3_{cfg[\'objects\'][b]}/object"','f"{cfg[\'objects\'][b]}/object"')
    text=text.replace("sys.argv=[sys.argv[0],", "from legacy_source_frame_playback import install as install_source_clock\ninstall_source_clock()\nsys.argv=[sys.argv[0],'task.source_frame_playback=true','scene.decompose_objects=false','sim.nconmax=2048','sim.njmax=4096','env.voc_scale=0.0','env.voc.max_force=0.0','env.voc.max_torque=0.0',")
    (root/'scripts/quality_optimize_grasps.py').write_text(text)
    _docker(['exec',native,'env','CUDA_VISIBLE_DEVICES=','JAX_PLATFORMS=cpu','V2D_GRASP_REFERENCE=track3_coherent_variant',
        'V2D_SOURCE_HAND_SIGMA=8.0','V2D_CONTACT_FIT_OUTPUT=research/quality-grasp-fits.json',
        'python','-u','/episode/scripts/run_geometry_cpu.py','/episode/scripts/quality_optimize_grasps.py'],root/'logs/quality-static-contact-fit.log')
    shutil.copy2(BASE/'apply_collision_transitions.py',root/'scripts/apply_collision_transitions.py')
    docker(['exec',retarget,'env','V2D_TRANSITION_REFERENCE=track3_coherent_variant','V2D_CONTACT_FIT_OUTPUT=research/quality-grasp-fits.json',
        'V2D_CONTACT_APPLY_OUTPUT=research/quality-contact-applied.json','/isaac-sim/python.sh','-u','/episode/scripts/apply_collision_transitions.py'],root/'logs/quality-apply-contact-fit.log')
    after=pq.ParquetFile(reference).read()
    columns=['fps','object_body_position','object_body_wxyz','object_body_names']
    assert before.select(columns).equals(after.select(columns)),'Contact fitting changed object motion or source clock'
    save(marker,dict(updated_utc=utc(),before_sha256=source_sha,after_sha256=digest(reference),
        object_motion_and_source_clock_preserved=True,dynamic_policy_grasp_success_not_claimed=True))
