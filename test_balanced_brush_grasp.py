"""Check an explicit bristle-tip compliance approximation before full-motion use."""
import os,json,shutil,traceback
from improvement_common import BASE,BATCH,save,utc
from batch_native import docker
R=BASE/'runs/episode_000030_newton_consistent_brush_20261021';A=BATCH/'improvements_20261021/activities/episode_000030-balanced-brace-grip.json'
def status(state,phase,**extra):save(A,{'episode':30,'pid':os.getpid(),'process_script':__file__.split('/')[-1],'updated_utc':utc(),'status':state,'phase':phase,'description':'Testing a firmer left-hand brace and stronger right wrist on explicit fine convex colliders with compliant bristle tips. Rigid handle, mass, inertia and45N limits are preserved.',**extra})
def main():
 shutil.copy2(BASE/'brush_bristle_compliance.py',R/'scripts/brush_bristle_compliance.py');(R/'scripts/search_compliant_brush_grasp.py').write_text("import runpy\nfrom brush_bristle_compliance import install\ninstall()\nrunpy.run_path('/episode/scripts/diagnose_refined_grasp_search.py',run_name='__main__')\n")
 for stiffness,damping in [(20.,1.)]:
  status('running',f'Physical hold/lift search with bristle-tip stiffness{stiffness:g}')
  label=f'compliant-brush-{int(stiffness)}-balanced-grasp-source89';profile={'stiffness':stiffness,'damping':damping,'tip_max_z':-.015}
  docker(['exec','v2d-consistent-episode30-newton','env','V2D_BRISTLE_COMPLIANCE='+json.dumps(profile),'V2D_BRISTLE_MATERIAL_REPORT=/episode/research/'+label+'-material.json','V2D_WRIST_MOTOR_PROFILE={"kp":500,"kd":30}','V2D_DIAGNOSTIC_WRIST_ANGULAR_KP=100','V2D_GRASP_WRIST_SIDE_PROFILES={"left":{"kp":100,"kd":12}}','V2D_GRASP_ANGULAR_SIDE_KP={"left":25,"right":100}','V2D_GRASP_SEARCH_GENERATIONS=12','V2D_STATIC_GRASP_SOURCE_FRAME=89','V2D_GRASP_PROBE_REFERENCE=track3_fine_collision_brush_candidate','V2D_REFINED_GRASP_LABEL='+label,'V2D_GRASP_SEED_FIT=evaluation/fine-brush-raised-grasp-source89.json','V2D_GRASP_SEARCH_OBJECTS=["mini_sweeper"]','V2D_GRASP_CONSISTENT_INITIAL_MOTORS=1','V2D_GRASP_SEARCH_REJECT_INVALID=1','V2D_PROTECT_OTHER_OBJECTS=1','V2D_GRASP_TRANSLATION_BOUND_M=.06','V2D_GRASP_ROTATION_BOUND_RAD=.8','V2D_GRASP_MIN_WRIST_Z_DELTA=.03','python','-u','/episode/scripts/search_compliant_brush_grasp.py'],R/f'logs/{label}.log')
  result=json.loads((R/f'evaluation/{label}.json').read_text());fit=result['best_candidates']['mini_sweeper'];save(R/f'research/{label}-fit.json',{'material_profile':profile,'fit':fit});print('Compliant tips',stiffness,'stable',fit['stable_hold_and_lift'],'cost',fit['cost'],flush=True)
  if fit['stable_hold_and_lift']:status('completed','Stable diagnostic grasp found; full-motion validation remains required',profile=profile,fit=fit);return
 status('completed','No tested bristle-tip material produced a stable held grasp; no policy promotion or training started',stable_hold_and_lift=False)
if __name__=='__main__':
 try:main()
 except Exception:status('failed','Compliant contact experiment failed; accepted video retained',error=traceback.format_exc());raise
