"""Rebuild and physically test isolated source-supported continuous grip candidates."""
import hashlib,json,shutil,subprocess,shlex
from improvement_common import BASE,save,utc
from batch_native import _docker,docker
R=BASE/'runs/episode_000016_newton_grasp_curriculum_20261020'
manifest=json.loads((R/'coherent-input-checksums.json').read_text())
def verify():
 for p,h in manifest.items():assert hashlib.sha256((R/p).read_bytes()).hexdigest()==h,p
verify();image=_docker(['inspect','v2d-batch-episode32-retarget','--format','{{.Image}}']).stdout.strip()
shutil.copy2(BASE/'apply_collision_transitions.py',R/'scripts/apply_collision_transitions.py');shutil.copy2(BASE/'run_geometry_cpu.py',R/'scripts/run_geometry_cpu.py')
for mm in [0,40]:
 ref=f'track3_continuous_right_slide{mm:03d}_probe'
 args=['run','--rm','--name',f'v2d-spoon16-continuous-{mm}-fk-cpu','--ipc','host','-e','CUDA_VISIBLE_DEVICES=',
  '-e',f'V2D_TRANSITION_REFERENCE={ref}','-e','V2D_REBUILD_ONLY=1','-e',f'V2D_CONTACT_APPLY_OUTPUT=research/{ref}-fk.json',
  '-v',f'{R}:/episode','-v',f'{BASE}/video_to_data/robotic_grounding/source/robotic_grounding/robotic_grounding/assets:/v2d/assets:ro',
  '--entrypoint','/isaac-sim/python.sh',image,'-u','/episode/scripts/apply_collision_transitions.py']
 _docker(args,R/f'logs/{ref}-fk.log');verify()
 _docker(['exec','v2d-grasp-episode16-newton','env','CUDA_VISIBLE_DEVICES=','JAX_PLATFORMS=cpu',f'V2D_AUDIT_REFERENCE={ref}',
  'python','-u','/episode/scripts/run_geometry_cpu.py','/episode/scripts/native_reference_audit.py'],R/f'logs/{ref}-collision.log')
 audit=json.loads((R/f'evaluation/{ref}-collision-audit.json').read_text())
 print(ref,'collision',audit['maximum_cross_body_penetration_mm'],'over10',audit['source_frames_over_10mm'],flush=True)
 docker(['exec','v2d-grasp-episode16-newton','env',f'V2D_GRASP_PROBE_REFERENCE={ref}','V2D_STATIC_GRASP_SOURCE_FRAME=74',
  'V2D_WRIST_MOTOR_PROFILE={"kp":25,"kd":6}','V2D_STATIC_GRASP_LIFT_SIDES=["right"]',
  'python','-u','/episode/scripts/diagnose_static_grasp.py'],R/f'logs/{ref}-static74.log')
 test=json.loads((R/f'evaluation/static-grasp-hold-lift-source-74-v2-kp25-kd6-reference-{ref}-lift-right.json').read_text())
 report=json.loads((R/f'research/{ref}.json').read_text());report.update(fk_pending=False,updated_utc=utc(),
  maximum_cross_body_penetration_mm=audit['maximum_cross_body_penetration_mm'],source_frames_over_10mm=audit['source_frames_over_10mm'],static_grasp_objects=test['objects'])
 save(R/f'research/{ref}.json',report);print(ref,test['objects'],flush=True)
verify()
