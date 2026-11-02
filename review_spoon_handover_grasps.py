"""Check whether each side of the observed handover can physically hold the spoon."""
import shutil,json
from improvement_common import BASE,save,utc
from batch_native import docker
R=BASE/'runs/episode_000016_newton_grasp_curriculum_20261020'
shutil.copy2(BASE/'diagnose_static_grasp.py',R/'scripts/diagnose_static_grasp.py')
results=[]
for source,side in [(65,'left'),(67,'right')]:
 docker(['exec','v2d-grasp-episode16-newton','env',f'V2D_STATIC_GRASP_SOURCE_FRAME={source}',
  'V2D_WRIST_MOTOR_PROFILE={"kp":25,"kd":6}',f'V2D_STATIC_GRASP_LIFT_SIDES={json.dumps([side])}',
  'python','-u','/episode/scripts/diagnose_static_grasp.py'],R/f'logs/handover-static-source{source}.log')
 p=R/f'evaluation/static-grasp-hold-lift-source-{source}-v2-kp25-kd6-lift-{side}.json';result=json.loads(p.read_text());results.append(result)
 print(source,side,result['objects'],flush=True)
save(R/'research/handover-grasp-physics-review.json',{'updated_utc':utc(),'diagnostic_only':True,'policy_inference':False,'object_assistance':0.,'tests':results})
