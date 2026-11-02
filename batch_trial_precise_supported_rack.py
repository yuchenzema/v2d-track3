"""Test whether finer released-rack collision geometry preserves the closer video fit."""
import copy,fcntl,hashlib,json,shutil,subprocess,time,xml.etree.ElementTree as ET
from batch_worker import BASE,PY,context
from batch_native import containers,docker
r,c=context(14);donor,_=context(15)
with (r/'research/collision-reference.lock').open('a') as lock:
 fcntl.flock(lock,fcntl.LOCK_EX);assert not (r/'coherent-input-checksums.json').exists();decision=json.loads((r/'research/terminal-support-reference-selection.json').read_text());assert decision['accepted'];reference=c['training_reference'];candidate='track3_precise_rack_original_support';shutil.copytree(r/f'human_motion_data/{reference}',r/f'human_motion_data/{candidate}',dirs_exist_ok=True);candidate_path=next((r/f'human_motion_data/{candidate}/processed').rglob('motion.parquet'));shutil.copy2(r/decision['backup']/'motion.parquet',candidate_path);mesh=r/'reconstruction/mesh_priors';donor_mesh=donor/'reconstruction/mesh_priors';assert hashlib.sha256((mesh/'plastic_dish_rack.obj').read_bytes()).digest()==hashlib.sha256((donor_mesh/'plastic_dish_rack.obj').read_bytes()).digest();backup=r/f'research/precise-supported-rack-{time.time_ns()}';backup.mkdir();urdf=mesh/'plastic_dish_rack.urdf';shutil.copy2(urdf,backup/'plastic_dish_rack.urdf');shutil.copy2(r/'evaluation/object-collision-hulls.npz',backup/'object-collision-hulls.npz');tree=ET.parse(urdf);link=tree.find('.//link');old=link.findall('collision')
 for node in old:link.remove(node)
 collisions=ET.parse(donor_mesh/'plastic_dish_rack.urdf').findall('.//collision')
 for node in collisions:link.append(copy.deepcopy(node))
 for part in donor_mesh.glob('plastic_dish_rack_precise_*.obj'):shutil.copy2(part,mesh/part.name)
 tree.write(urdf,encoding='unicode');native,_=containers(r,14)
 for name in ['native_reference_audit','diagnose_terminal_support']:shutil.copy2(BASE/f'{name}.py',r/f'scripts/{name}.py')
 trials=[];accepted=False
 try:
  docker(['exec',native,'env',f'V2D_AUDIT_REFERENCE={candidate}','python','-u','/episode/scripts/native_reference_audit.py'],r/'logs/precise-supported-rack-audit.log');audit=json.loads((r/f'evaluation/{candidate}-collision-audit.json').read_text())
  for steps in [10,20]:
   label=f'precise-rack-original-terminal-{20*steps}hz';docker(['exec',native,'env',f'V2D_DIAGNOSTIC_REFERENCE={candidate}',f'V2D_DIAGNOSTIC_SUBSTEPS={steps}',f'V2D_DIAGNOSTIC_LABEL={label}','python','-u','/episode/scripts/diagnose_terminal_support.py'],r/f'logs/{label}.log');test=json.loads((r/f'evaluation/{label}.json').read_text());trials.append(test);target=next(x for x in test['objects'] if x['object']=='blue_cup');accepted=test['all_states_finite'] and target['maximum_position_drift_m']<=.015 and target['maximum_rotation_drift_rad']<=.2 and audit['source_frames_over_10mm']<=2 and audit['maximum_cross_body_penetration_mm']<=12
   print('Precise rack original video pose',20*steps,target,'accepted',accepted,flush=True)
   if accepted:
    path=next((r/f'human_motion_data/{reference}/processed').rglob('motion.parquet'));shutil.copy2(path,backup/'coarse-supported-motion.parquet');temp=path.with_suffix('.precise.tmp');shutil.copy2(candidate_path,temp);temp.replace(path);audit['reference']=reference;p=r/f'evaluation/{reference}-collision-audit.json';temp=p.with_suffix('.precise.tmp');temp.write_text(json.dumps(audit,indent=2));temp.replace(p);cfg=json.loads((r/'episode-config.json').read_text());cfg.update(physics_substeps=steps,physics_integration_hz=steps*20,physics_integration_reason='Finer released-rack collider supports the original source placement; checked independently under gravity.');(r/'episode-config.json').write_text(json.dumps(cfg,indent=2));subprocess.run([PY,str(BASE/'assess_physical_reference.py'),'--run',str(r),'--reference',reference,'--overlay'],check=True);decision.update(precise_rack_original_pose_accepted=True,terminal_support_test=test,maximum_cross_body_penetration_mm=audit['maximum_cross_body_penetration_mm'],source_frames_over_10mm=audit['source_frames_over_10mm'],source_fidelity_after=json.loads((r/'evaluation/physical-reference-source-fidelity.json').read_text())['aggregate']);(r/'research/terminal-support-reference-selection.json').write_text(json.dumps(decision,indent=2));break
 finally:
  if not accepted:
   for source,dest in [(backup/'plastic_dish_rack.urdf',urdf),(backup/'object-collision-hulls.npz',r/'evaluation/object-collision-hulls.npz')]:
    temp=dest.with_suffix('.restore.tmp');shutil.copy2(source,temp);temp.replace(dest)
 report={'method':__doc__,'accepted':accepted,'rack_visual_mesh_and_mass_preserved':True,'fine_hull_count':len(collisions),'source_pose_reference':candidate,'trials':trials,'backup':str(backup.relative_to(r))};(r/'research/precise-supported-rack-selection.json').write_text(json.dumps(report,indent=2));print(report,flush=True)
