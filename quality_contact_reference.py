"""Fit new own-geometry contact targets before freezing a quality experiment."""
import json,shutil
from pathlib import Path
import pyarrow.parquet as pq
from improvement_common import BASE,save,utc,digest
from batch_native import _docker,docker


def optimizer_script():
    text=(BASE/'optimize_native_grasps.py').read_text()
    # The penetration-only objective allowed 0.5 mm overlap, while the
    # supporting-contact acceptance correctly required clearance. Give both
    # objectives the same positive clearance target; keep acceptance unchanged.
    allowance='contact_allowance_m=-.0005 if strict_contacts else .0005'
    assert allowance in text
    text=text.replace(allowance,'contact_allowance_m=-.0005')
    text=text.replace('results=[];O=',"results=[]\n import re\n for values in names.values():\n  for label,index in list(values.items()):\n   if label:values[re.sub(r'[^a-zA-Z0-9_]', '_', label)]=index\n O=")
    text=text.replace('f"track3_{cfg[\'objects\'][b]}/object"','f"{cfg[\'objects\'][b]}/object"')
    original="  target=tip_positions();active=np.array(c[f'mano_{side}_object_contact_part_ids'])[t]>0"
    proposal="""  target=tip_positions();active=np.array(c[f'mano_{side}_object_contact_part_ids'])[t]>0
  if not frame_mode:
   import trimesh
   surface=trimesh.load(R/f"reconstruction/mesh_priors/{cfg['objects'][b]}.ply",force='mesh',process=False)
   object_rotation=Rotation.from_quat(OQ[t,b],scalar_first=True).as_matrix()
   vertices=surface.vertices@object_rotation.T+O[t,b];normals=surface.vertex_normals@object_rotation.T
   opening=target[0]-target[1:].mean(0);opening/=np.linalg.norm(opening)+1e-12
   for tip in range(5):
    allowed=np.flatnonzero((np.linalg.norm(normals,axis=1)>.5)&((normals@opening>.1) if tip==0 else (normals@opening<-.1)))
    if not len(allowed):continue
    nearest=allowed[np.argmin(np.linalg.norm(vertices[allowed]-target[tip],axis=1))]
    if np.linalg.norm(vertices[nearest]-target[tip])<=.04 and (active[tip] or tip in [0,1,2]):
     target[tip]=vertices[nearest]+normals[nearest]*.0008;active[tip]=True
"""
    assert original in text;text=text.replace(original,proposal)
    return text.replace("sys.argv=[sys.argv[0],", "from legacy_source_frame_playback import install as install_source_clock\ninstall_source_clock()\nsys.argv=[sys.argv[0],'task.source_frame_playback=true','scene.decompose_objects=false','sim.nconmax=2048','sim.njmax=4096','env.voc_scale=0.0','env.voc.max_force=0.0','env.voc.max_torque=0.0',")


def fit(root,native,retarget):
    root=Path(root);cfg=json.loads((root/'episode-config.json').read_text());marker=root/'research/quality-contact-reference.json'
    if marker.exists():return
    assert not (root/'coherent-input-checksums.json').exists(),'Keep trained references immutable'
    reference=root/f"human_motion_data/track3_coherent_variant/processed/sequence_id={cfg['sequence']}/robot_name=sharpa_wave/motion.parquet"
    before=pq.ParquetFile(reference).read();source_sha=digest(reference)
    for name in ['native_pinch_contacts.py','grasp_physics_checks.py','rigid_pose_collision_audit.py','run_geometry_cpu.py']:
        shutil.copy2(BASE/name,root/'scripts'/name)
    text=optimizer_script()
    (root/'scripts/quality_optimize_grasps.py').write_text(text)
    _docker(['exec',native,'env','CUDA_VISIBLE_DEVICES=','JAX_PLATFORMS=cpu','V2D_GRASP_REFERENCE=track3_coherent_variant',
        'V2D_SOURCE_HAND_SIGMA=8.0','V2D_CONTACT_FIT_OUTPUT=research/quality-grasp-fits.json',
        'python','-u','/episode/scripts/run_geometry_cpu.py','/episode/scripts/quality_optimize_grasps.py'],root/'logs/quality-static-contact-fit.log')
    application=(BASE/'apply_collision_transitions.py').read_text().replace('lo=start;hi=end;localbase=', 'lo=max(0,start-5);hi=min(T-1,end+4);localbase=')
    (root/'scripts/apply_collision_transitions.py').write_text(application)
    docker(['exec',retarget,'env','V2D_TRANSITION_REFERENCE=track3_coherent_variant','V2D_CONTACT_FIT_OUTPUT=research/quality-grasp-fits.json',
        'V2D_CONTACT_APPLY_OUTPUT=research/quality-contact-applied.json','/isaac-sim/python.sh','-u','/episode/scripts/apply_collision_transitions.py'],root/'logs/quality-apply-contact-fit.log')
    after=pq.ParquetFile(reference).read()
    columns=['fps','object_body_position','object_body_wxyz','object_body_names']
    assert before.select(columns).equals(after.select(columns)),'Contact fitting changed object motion or source clock'
    save(marker,dict(updated_utc=utc(),before_sha256=source_sha,after_sha256=digest(reference),
        object_motion_and_source_clock_preserved=True,dynamic_policy_grasp_success_not_claimed=True,
        optimizer_script_sha256=digest(root/'scripts/quality_optimize_grasps.py'),
        optimizer_revision='supporting_contact_clearance_v2'))
