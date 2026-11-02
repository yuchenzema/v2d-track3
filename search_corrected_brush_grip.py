"""Refit the brush grip away from the observed thumb/rack intersection."""
import os,json,traceback,shutil
from improvement_common import BASE,BATCH,save,utc
from batch_native import docker
R=BASE/'runs/episode_000030_newton_consistent_brush_20261021';A=BATCH/'improvements_20261021/activities/episode_000030-corrected-grip-search.json'
def status(state,phase,**extra):save(A,{'episode':30,'pid':os.getpid(),'process_script':__file__.split('/')[-1],'updated_utc':utc(),'status':state,'phase':phase,'description':'Refitting the brush grip to avoid the robot thumb intersecting the rack during the long hold. Physical diagnostic; current video retained.',**extra})
def main():
 status('running','Six generations of physical hold/lift grasp candidates at source89')
 text=(BASE/'diagnose_static_grasp.py').read_text().replace("'sim.njmax=1536'","f\"sim.njmax={cfg.get('physics_njmax',1536)}\",f\"sim.nconmax={cfg.get('physics_nconmax',512)}\"")
 (R/'scripts/diagnose_static_grasp.py').write_text(text);shutil.copy2(BASE/'diagnose_refined_grasp_search.py',R/'scripts/diagnose_refined_grasp_search.py')
 label='corrected-branch-grasp-search-source89'
 docker(['exec','v2d-consistent-episode30-newton','env','V2D_STATIC_GRASP_SOURCE_FRAME=89','V2D_REFINED_GRASP_LABEL='+label,'V2D_GRASP_USE_REFERENCE_SEED=1','V2D_GRASP_SEARCH_OBJECTS=["mini_sweeper"]','V2D_GRASP_CONSISTENT_INITIAL_MOTORS=1','V2D_GRASP_SEARCH_REJECT_INVALID=1','V2D_PROTECT_OTHER_OBJECTS=1','V2D_GRASP_TRANSLATION_BOUND_M=.045','V2D_GRASP_ROTATION_BOUND_RAD=.6','python','-u','/episode/scripts/diagnose_refined_grasp_search.py'],R/'logs/corrected-branch-grasp-search-source89.log')
 d=json.loads((R/f'evaluation/{label}.json').read_text());save(R/'research/corrected-branch-grasp-search-summary.json',d);status('completed','Corrected-branch grip search finished; independent motion validation still required',result=d['best_candidates']['mini_sweeper']);print(json.dumps(d['best_candidates']['mini_sweeper']),flush=True)
if __name__=='__main__':
 try:main()
 except Exception:status('failed','Grip search failed; retained video unchanged',error=traceback.format_exc());raise
