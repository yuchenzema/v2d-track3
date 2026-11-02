"""Audit and evaluate an isolated bimanual support correction against the source video."""
import hashlib,json,shutil,subprocess
from improvement_common import BASE,save,utc
from batch_native import _docker,docker
from batch_review import review
R=BASE/'runs/episode_000016_newton_grasp_curriculum_20261020';ref='track3_supported_handover_pair_probe'
report=json.loads((R/'research/supported-handover-pair-probe.json').read_text());manifest=json.loads((R/'coherent-input-checksums.json').read_text())
def verify():
 for p,h in manifest.items():assert hashlib.sha256((R/p).read_bytes()).hexdigest()==h,p
verify();shutil.copy2(BASE/'apply_collision_transitions.py',R/'scripts/apply_collision_transitions.py');image=_docker(['inspect','v2d-batch-episode32-retarget','--format','{{.Image}}']).stdout.strip()
_docker(['run','--rm','--name','v2d-spoon16-pair-fk-cpu','--ipc','host','-e','CUDA_VISIBLE_DEVICES=',
 '-e',f'V2D_TRANSITION_REFERENCE={ref}','-e','V2D_REBUILD_ONLY=1','-e','V2D_CONTACT_APPLY_OUTPUT=research/supported-handover-pair-fk.json',
 '-e','V2D_CONTACT_GRASPS_OVERRIDE='+json.dumps(report['human_contact_annotations']),
 '-v',f'{R}:/episode','-v',f'{BASE}/video_to_data/robotic_grounding/source/robotic_grounding/robotic_grounding/assets:/v2d/assets:ro',
 '--entrypoint','/isaac-sim/python.sh',image,'-u','/episode/scripts/apply_collision_transitions.py'],R/'logs/supported-handover-pair-fk.log')
verify();_docker(['exec','v2d-grasp-episode16-newton','env','CUDA_VISIBLE_DEVICES=','JAX_PLATFORMS=cpu',f'V2D_AUDIT_REFERENCE={ref}',
 'python','-u','/episode/scripts/run_geometry_cpu.py','/episode/scripts/native_reference_audit.py'],R/'logs/supported-handover-pair-collision.log')
audit=json.loads((R/f'evaluation/{ref}-collision-audit.json').read_text());report.update(fk_pending=False,updated_utc=utc(),maximum_cross_body_penetration_mm=audit['maximum_cross_body_penetration_mm'],source_frames_over_10mm=audit['source_frames_over_10mm'])
save(R/'research/supported-handover-pair-probe.json',report)
print('Handover pair collision audit',report['maximum_cross_body_penetration_mm'],report['source_frames_over_10mm'],flush=True)
if not audit['source_frames_over_10mm']:
 for source in [67,103]:
  docker(['exec','v2d-grasp-episode16-newton','env',f'V2D_GRASP_PROBE_REFERENCE={ref}',f'V2D_STATIC_GRASP_SOURCE_FRAME={source}',
   'V2D_WRIST_MOTOR_PROFILE={"kp":25,"kd":6}','V2D_STATIC_GRASP_LIFT_SIDES=["right"]','python','-u','/episode/scripts/diagnose_static_grasp.py'],R/f'logs/supported-handover-pair-static{source}.log')
  p=R/f'evaluation/static-grasp-hold-lift-source-{source}-v2-kp25-kd6-reference-{ref}-lift-right.json';print(source,json.loads(p.read_text())['objects'],flush=True)
 step=360000000;row=review(16,step,reference_override=ref,substeps_override=20,root_override=R,container_override='v2d-grasp-episode16-newton')
 save(R/'research/supported-handover-pair-policy-review.json',row);print('Pair policy review',{k:row[k] for k in ['source_mean_object_iou','source_worst_phase_object_iou','completion_rate_32_worlds']},flush=True)
verify()
