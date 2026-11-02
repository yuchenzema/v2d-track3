"""Check the corrected brush grasp at pickup, extraction and the long hold."""
import os,json,traceback
from improvement_common import BASE,BATCH,save,utc
from batch_native import docker
R=BASE/'runs/episode_000030_newton_consistent_brush_20261021'
A=BATCH/'improvements_20261021/activities/episode_000030-grip-stability.json'
def status(state,phase,**extra):save(A,{'episode':30,'pid':os.getpid(),'process_script':__file__.split('/')[-1],'updated_utc':utc(),'status':state,'phase':phase,'description':'Testing whether the reconstructed brush grip holds and lifts at pickup, extraction and pause. Diagnostic only; current video unchanged.',**extra})
def main():
 text=(BASE/'diagnose_static_grasp.py').read_text().replace("'sim.njmax=1536'","f\"sim.njmax={cfg.get('physics_njmax',1536)}\",f\"sim.nconmax={cfg.get('physics_nconmax',512)}\"")
 (R/'scripts/diagnose_corrected_brush_static_grasp.py').write_text(text)
 results=[]
 for frame in [34,65,89]:
  status('running',f'Static hold and3cm lift at source frame{frame}')
  docker(['exec','v2d-consistent-episode30-newton','env',f'V2D_STATIC_GRASP_SOURCE_FRAME={frame}','V2D_WRIST_MOTOR_PROFILE={"kp":250,"kd":20}','V2D_WRIST_SIDE_MOTOR_PROFILES={"left":{"kp":25,"kd":6}}','V2D_STATIC_GRASP_LIFT_SIDES=["right"]','python','-u','/episode/scripts/diagnose_corrected_brush_static_grasp.py'],R/f'logs/corrected-brush-grip-source{frame}.log')
  p=R/f'evaluation/static-grasp-hold-lift-source-{frame}-v2-kp250-kd20-lift-right.json';d=json.loads(p.read_text());results.append({'source_frame':frame,**d['objects']['mini_sweeper']});print('Brush static grip',results[-1],flush=True)
 save(R/'research/corrected-brush-grip-stability.json',{'updated_utc':utc(),'results':results,'all_phases_stable':all(x['stable_hold_and_lift'] for x in results)});status('completed','Brush grip stability checks finished',results=results)
if __name__=='__main__':
 try:main()
 except Exception:status('failed','Brush diagnostic failed; current video unchanged',error=traceback.format_exc());raise
