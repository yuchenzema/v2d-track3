"""Compare an isolated gravity-registered reference using real policy rollouts."""
from pathlib import Path
import hashlib,json,subprocess,time,shutil
from batch_native import _docker
from batch_review import review
from improvement_common import BASE,save,utc
R=BASE/'runs/episode_000016_newton_grasp_curriculum_20261020';ref='track3_gravity_supported_probe'
manifest=json.loads((R/'coherent-input-checksums.json').read_text())
def verify():
 for p,h in manifest.items():assert hashlib.sha256((R/p).read_bytes()).hexdigest()==h,p
verify()
assert (R/'research/gravity-probe-fk-rebuilt.json').exists()
shutil.copy2(BASE/'run_geometry_cpu.py',R/'scripts/run_geometry_cpu.py')
_docker(['exec','v2d-grasp-episode16-newton','env','CUDA_VISIBLE_DEVICES=','JAX_PLATFORMS=cpu',f'V2D_AUDIT_REFERENCE={ref}','python','-u','/episode/scripts/run_geometry_cpu.py','/episode/scripts/native_reference_audit.py'],R/'logs/gravity-probe-collision-audit.log')
audit=json.loads((R/f'evaluation/{ref}-collision-audit.json').read_text())
subprocess.run([str(R/'.venv-video/bin/python'),str(BASE/'assess_physical_reference.py'),'--run',str(R),'--reference',ref,'--output','gravity-probe-reference-source-fidelity','--no-update-selection'],check=True)
report=json.loads((R/'research/gravity-registered-rack-probe.json').read_text());report.update(updated_utc=utc(),fk_and_collision_audit_pending=False,
 maximum_cross_body_penetration_mm=audit['maximum_cross_body_penetration_mm'],source_frames_over_10mm=audit['source_frames_over_10mm'])
if audit['source_frames_over_10mm']:
 report['rejected_reason']='The registered insertion path needs collision refinement before policy comparison.'
else:
 rows=[]
 for step,cp in [(355000320,'/episode/research/handover-schedule-checkpoint-355000320/policy_355000320.safetensors'),(350007296,'/episode/research/initial-policy/policy_350007296.safetensors')]:
  row=review(16,step,reference_override=ref,substeps_override=20,checkpoint_override=cp,root_override=R,container_override='v2d-grasp-episode16-newton')
  save(R/f'research/gravity-registered-policy-{step}-review.json',row);rows.append(row)
  print(step,{k:row[k] for k in ['source_mean_object_iou','completion_rate_32_worlds','source_worst_phase_object_iou']},flush=True)
 report['policy_reviews']=[f'research/gravity-registered-policy-{r["environment_steps"]}-review.json' for r in rows]
verify();save(R/'research/gravity-registered-rack-probe.json',report)
print('Gravity registration probe finished. Active training inputs unchanged; no candidate auto-promoted.',flush=True)
