"""Check and evaluate the previously stable insertion grip with late-only impedance."""
import hashlib,json,shutil
from improvement_common import BASE,save,utc
from batch_native import _docker
from batch_review import review
R=BASE/'runs/episode_000016_newton_grasp_curriculum_20261020';ref='track3_late_high_grip_probe';frozen=json.loads((R/'coherent-input-checksums.json').read_text())
def verify():
 for p,h in frozen.items():assert hashlib.sha256((R/p).read_bytes()).hexdigest()==h,p
verify();shutil.copy2(BASE/'apply_collision_transitions.py',R/'scripts/apply_collision_transitions.py');image=_docker(['inspect','v2d-batch-episode32-retarget','--format','{{.Image}}']).stdout.strip()
_docker(['run','--rm','--name','v2d-spoon16-late-high-fk-cpu','--ipc','host','-e','CUDA_VISIBLE_DEVICES=',
 '-e',f'V2D_TRANSITION_REFERENCE={ref}','-e','V2D_REBUILD_ONLY=1','-e','V2D_CONTACT_APPLY_OUTPUT=research/late-high-grip-fk.json',
 '-v',f'{R}:/episode','-v',f'{BASE}/video_to_data/robotic_grounding/source/robotic_grounding/robotic_grounding/assets:/v2d/assets:ro',
 '--entrypoint','/isaac-sim/python.sh',image,'-u','/episode/scripts/apply_collision_transitions.py'],R/'logs/late-high-grip-fk.log')
verify();_docker(['exec','v2d-grasp-episode16-newton','env','CUDA_VISIBLE_DEVICES=','JAX_PLATFORMS=cpu',f'V2D_AUDIT_REFERENCE={ref}',
 'python','-u','/episode/scripts/run_geometry_cpu.py','/episode/scripts/native_reference_audit.py'],R/'logs/late-high-grip-collision.log')
audit=json.loads((R/f'evaluation/{ref}-collision-audit.json').read_text());report=json.loads((R/'research/late-high-grip-probe.json').read_text());report.update(fk_pending=False,updated_utc=utc(),maximum_cross_body_penetration_mm=audit['maximum_cross_body_penetration_mm'],source_frames_over_10mm=audit['source_frames_over_10mm']);save(R/'research/late-high-grip-probe.json',report)
print('Late high grip collisions',report['maximum_cross_body_penetration_mm'],report['source_frames_over_10mm'],flush=True)
if not audit['source_frames_over_10mm']:
 profile={'right':{'kp':250.,'kd':20.,'start_source_frame':84,'ramp_source_frames':16}}
 row=review(16,360000000,reference_override=ref,substeps_override=20,root_override=R,container_override='v2d-grasp-episode16-newton',timed_impedance=profile)
 save(R/'research/late-high-grip-timed-impedance-policy-review.json',row);print('Late high-grip policy',{k:row[k] for k in ['source_mean_object_iou','source_worst_phase_object_iou','completion_rate_32_worlds']},flush=True)
verify()
