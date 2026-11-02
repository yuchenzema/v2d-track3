"""Test finer brush collision geometry and a raised, rack-clear grasp before learning."""
import os,json,copy,shutil,subprocess,traceback,xml.etree.ElementTree as ET
import numpy as np,pyarrow as pa,pyarrow.parquet as pq,trimesh
from scipy.spatial.transform import Rotation,Slerp
from improvement_common import BASE,BATCH,save,utc,digest,better
from batch_native import _docker,docker
from batch_review import review
from video_catalog import catalog
R=BASE/'runs/episode_000030_newton_consistent_brush_20261021';REF='track3_fine_collision_brush_candidate';A=BATCH/'improvements_20261021/activities/episode_000030-fine-collision.json'
def status(state,phase,**extra):save(A,{'episode':30,'pid':os.getpid(),'process_script':__file__.split('/')[-1],'updated_utc':utc(),'status':state,'phase':phase,'description':'Testing finer brush colliders and a raised grip to clear the rack. Public visual mesh, mass and inertia are preserved; current video remains available.',**extra})
def main():
 status('running','Refining convex brush collision pieces from the unchanged public CAD')
 script="""from pathlib import Path
import json,time,trimesh,numpy as np
from flash_chord.assets.mesh import convex_decompose
r=Path('/episode');m=trimesh.load(r/'reconstruction/mesh_priors/mini_sweeper.obj',force='mesh',process=False);start=time.time();hulls=convex_decompose(m.vertices,m.faces,threshold=.01);arrays={}
for i,h in enumerate(hulls):arrays[f'v{i}']=h.vertices;arrays[f'f{i}']=h.faces
np.savez_compressed(r/'research/brush-precise-convex-hulls.npz',**arrays)
(r/'research/brush-precise-convex-hulls.json').write_text(json.dumps({'threshold':.01,'hull_count':len(hulls),'elapsed_s':time.time()-start,'visual_geometry_unchanged':True},indent=2));print('Fine brush hulls',len(hulls),flush=True)
"""
 (R/'scripts/decompose_brush_precise.py').write_text(script)
 _docker(['exec','v2d-consistent-episode30-newton','env','CUDA_VISIBLE_DEVICES=','JAX_PLATFORMS=cpu','python','-u','/episode/scripts/decompose_brush_precise.py'],R/'logs/decompose-brush-precise.log')
 geometry=R/'research/fine-brush-geometry';geometry.mkdir(exist_ok=True);source=R/'reconstruction/mesh_priors/mini_sweeper.urdf';tree=ET.parse(source);link=tree.find('.//link');inertia=ET.tostring(link.find('inertial'));old=copy.deepcopy(link.find('collision'));origin=old.find('origin');scale=old.find('geometry/mesh').get('scale')
 for node in list(link.findall('collision')):link.remove(node)
 with np.load(R/'research/brush-precise-convex-hulls.npz') as hulls:
  count=len(hulls.files)//2
  for i in range(count):
   file=geometry/f'brush-convex-{i:03d}.obj';trimesh.Trimesh(hulls[f'v{i}'],hulls[f'f{i}'],process=False).export(file);node=ET.SubElement(link,'collision')
   if origin is not None:node.append(copy.deepcopy(origin))
   mesh=ET.SubElement(ET.SubElement(node,'geometry'),'mesh',filename='/episode/'+str(file.relative_to(R)))
   if scale:mesh.set('scale',scale)
 assert ET.tostring(link.find('inertial'))==inertia
 target_urdf=geometry/'mini_sweeper.urdf';tree.write(target_urdf,encoding='unicode')
 save(geometry/'provenance.json',{'updated_utc':utc(),'collision_hulls':count,'original_visual_cad_sha256':digest(R/'reconstruction/mesh_priors/mini_sweeper.obj'),'original_urdf_sha256':digest(source),'refined_urdf_sha256':digest(target_urdf),'visual_geometry_and_inertia_unchanged':True,'original_training_assets_unchanged':True})
 target=R/'human_motion_data'/REF;assert not target.exists();shutil.copytree(R/'human_motion_data/track3_consistent_brush_pickup_candidate',target);path=next(target.rglob('motion.parquet'));row=pq.ParquetFile(path).read().to_pylist()[0];row['object_urdf_paths'][0]='/episode/'+str(target_urdf.relative_to(R));pq.write_table(pa.Table.from_pylist([row]),path,compression='zstd')
 status('running',f'Physical grasp search on{count} finer brush hulls; palm at least3cm above the old pose')
 label='fine-brush-raised-grasp-source89';docker(['exec','v2d-consistent-episode30-newton','env','V2D_STATIC_GRASP_SOURCE_FRAME=89','V2D_GRASP_PROBE_REFERENCE='+REF,'V2D_REFINED_GRASP_LABEL='+label,'V2D_GRASP_USE_REFERENCE_SEED=1','V2D_GRASP_SEARCH_OBJECTS=["mini_sweeper"]','V2D_GRASP_CONSISTENT_INITIAL_MOTORS=1','V2D_GRASP_SEARCH_REJECT_INVALID=1','V2D_PROTECT_OTHER_OBJECTS=1','V2D_GRASP_TRANSLATION_BOUND_M=.06','V2D_GRASP_ROTATION_BOUND_RAD=.8','V2D_GRASP_MIN_WRIST_Z_DELTA=.03','python','-u','/episode/scripts/diagnose_refined_grasp_search.py'],R/'logs/fine-brush-raised-grasp-search.log')
 fit=json.loads((R/f'evaluation/{label}.json').read_text())['best_candidates']['mini_sweeper'];save(R/'research/fine-brush-raised-grasp-fit.json',fit)
 if not fit['stable_hold_and_lift']:
  status('completed','Grasp failed the physical hold/lift checks; no training or publication started',stable_grasp_found=False,fit=fit);print('Finer brush collision grip still unstable',fit['cost'],flush=True);return
 status('running','Rebuilding a continuous reference with the independently tested raised grip')
 local=np.asarray(fit['wrist_in_object']);finger=np.asarray(fit['finger_joints']);P=np.asarray(row['object_body_position']);Q=np.asarray(row['object_body_wxyz']);w=np.asarray(row['robot_right_wrist_position']);q=np.asarray(row['robot_right_wrist_wxyz']);f=np.asarray(row['robot_right_finger_joints']);T=len(P)
 for i in range(26,254):
  O=np.eye(4);O[:3,:3]=Rotation.from_quat(Q[i,0],scalar_first=True).as_matrix();O[:3,3]=P[i,0];H=O@local;weight=min(float(np.clip((i-26)/8,0,1)),float(np.clip((253-i)/8,0,1)));weight=weight*weight*(3-2*weight);w[i]=(1-weight)*w[i]+weight*H[:3,3]
  if i<34:w[i,2]+=.04*np.sin(np.pi*(i-26)/8)
  q[i]=Slerp([0,1],Rotation.from_matrix([Rotation.from_quat(q[i],scalar_first=True).as_matrix(),H[:3,:3]]))([weight]).as_quat(scalar_first=True)[0];closure=min(float(np.clip((i-31)/3,0,1)),float(np.clip((250-i)/5,0,1)));closure=closure*closure*(3-2*closure);f[i]=(1-closure)*f[i]+closure*finger
 row['robot_right_wrist_position']=w.tolist();row['robot_right_wrist_wxyz']=q.tolist();row['robot_right_finger_joints']=f.tolist();pq.write_table(pa.Table.from_pylist([row]),path,compression='zstd')
 image=_docker(['inspect','v2d-batch-episode32-retarget','--format','{{.Image}}']).stdout.strip()
 _docker(['run','--rm','--name','v2d-fine-brush-fk-cpu','--ipc','host','-e','CUDA_VISIBLE_DEVICES=','-e','V2D_TRANSITION_REFERENCE='+REF,'-e','V2D_REBUILD_ONLY=1','-e','V2D_CONTACT_APPLY_OUTPUT=research/fine-brush-fk.json','-v',f'{R}:/episode','-v',f'{BASE}/video_to_data/robotic_grounding/source/robotic_grounding/robotic_grounding/assets:/v2d/assets:ro','--entrypoint','/isaac-sim/python.sh',image,'-u','/episode/scripts/apply_collision_transitions.py'],R/'logs/fine-brush-fk.log')
 status('running','Independent whole-source32-world rollout; no object assistance')
 profile={'left':0.,'right':.5,'maximum_correction_m':.015,'right_wrist_residual_gain':0.,'right_finger_residual_gain':0.}
 candidate=review(30,45001728,reference_override=REF,root_override=R,container_override='v2d-consistent-episode30-newton',substeps_override=20,recording_world_count=32,tracking_feedback=profile);previous=catalog(30,BASE/'runs/episode_000030_newton_batch_20261015',True)['best']['selected'];eligible=better(candidate,previous);candidate['passes_current_accepted_policy_gate']=eligible;save(R/'research/fine-brush-policy-review.json',candidate);status('completed','Finer collision and raised-grip policy review complete',eligible_for_independent_publication=eligible,source_mean_object_iou=candidate['source_mean_object_iou'],completion_rate=candidate['completion_rate_32_worlds']);print('Fine brush policy',candidate['source_mean_object_iou'],candidate['completion_rate_32_worlds'],'better',eligible,flush=True)
if __name__=='__main__':
 try:main()
 except Exception:status('failed','Finer collision experiment failed; retained video unchanged',error=traceback.format_exc());raise
