"""Test stronger robot wrist control with the original45N/Nm actuator limits."""
import os,json,traceback
from improvement_common import BASE,BATCH,save,utc
from batch_native import docker
R=BASE/'runs/episode_000030_newton_consistent_brush_20261021';A=BATCH/'improvements_20261021/activities/episode_000030-wrist-calibration.json'
def status(state,phase,**extra):save(A,{'episode':30,'pid':os.getpid(),'process_script':__file__.split('/')[-1],'updated_utc':utc(),'status':state,'phase':phase,'description':'Testing stronger robot wrist gains on the finer brush geometry. Actuator limits, object mass and inertia remain unchanged.',**extra})
def main():
 status('running','Physical grasp search with500N/m translation and100 angular wrist gain')
 label='fine-brush-stiffer-wrist-grasp-source89';docker(['exec','v2d-consistent-episode30-newton','env','V2D_STATIC_GRASP_SOURCE_FRAME=89','V2D_GRASP_PROBE_REFERENCE=track3_fine_collision_brush_candidate','V2D_REFINED_GRASP_LABEL='+label,'V2D_WRIST_MOTOR_PROFILE={"kp":500,"kd":30}','V2D_DIAGNOSTIC_WRIST_ANGULAR_KP=100','V2D_GRASP_SEED_FIT=evaluation/fine-brush-raised-grasp-source89.json','V2D_GRASP_SEARCH_OBJECTS=["mini_sweeper"]','V2D_GRASP_CONSISTENT_INITIAL_MOTORS=1','V2D_GRASP_SEARCH_REJECT_INVALID=1','V2D_PROTECT_OTHER_OBJECTS=1','V2D_GRASP_TRANSLATION_BOUND_M=.06','V2D_GRASP_ROTATION_BOUND_RAD=.8','V2D_GRASP_MIN_WRIST_Z_DELTA=.03','python','-u','/episode/scripts/diagnose_refined_grasp_search.py'],R/'logs/fine-brush-stiffer-wrist-grasp-search.log')
 result=json.loads((R/f'evaluation/{label}.json').read_text());fit=result['best_candidates']['mini_sweeper'];save(R/'research/fine-brush-stiffer-wrist-grasp-fit.json',fit);status('completed','Stronger wrist grasp calibration finished; full-motion policy validation remains separate',stable_hold_and_lift=fit['stable_hold_and_lift']);print('Stable stiffer wrist grip',fit['stable_hold_and_lift'],'cost',fit['cost'],'orientation',fit['object_orientation_change_rad'],'lift',fit['actual_hand_lift_m'],flush=True)
if __name__=='__main__':
 try:main()
 except Exception:status('failed','Wrist calibration failed; current video retained',error=traceback.format_exc());raise
