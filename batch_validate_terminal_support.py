"""Audit a supported reconstruction candidate before allowing policy training."""
import fcntl,json,os,shutil,subprocess,sys,time
from batch_worker import BASE,PY,context
from batch_native import containers,docker

def atomic_copy(source,destination):
 temp=destination.with_name(destination.name+'.support.tmp');shutil.copy2(source,temp);temp.replace(destination)

for ep in map(int,sys.argv[1:]):
 r,c=context(ep)
 while True:
  with (r/'research/collision-reference.lock').open('a') as lock:
   fcntl.flock(lock,fcntl.LOCK_EX)
   accepted_marker=r/'research/terminal-support-reference-selection.json'
   if os.environ.get('V2D_SKIP_ACCEPTED_SUPPORT')=='1' and accepted_marker.exists() and json.loads(accepted_marker.read_text()).get('accepted'):print(ep,'Keeping accepted supported placement',flush=True);break
   searchpath=r/'research/terminal-support-search.json'
   if searchpath.exists():
    search=json.loads(searchpath.read_text())
    # The superseded search used the source index for the resampled reference;
    # accept only the corrected final-frame initialization and full capacity.
    script=(r/'scripts/search_terminal_support.py').read_text()
    ready=search.get('search_version',0)>=int(os.environ.get('V2D_SUPPORT_SEARCH_MIN_VERSION','2')) and search.get('reference_final_frame')==2*(c['frames']-1)-1 and len(search.get('candidates',[]))==256
    if os.environ.get('V2D_SUPPORT_EXPECTED_OBJECT'):ready=ready and search.get('object')==os.environ['V2D_SUPPORT_EXPECTED_OBJECT']
   else:ready=False
   if not ready:pass
   else:
    assert not (r/'coherent-input-checksums.json').exists();best=search['best']
    if best is None:print(ep,'No stable supported pose found',flush=True);break
    reference=c['training_reference'];candidate='track3_terminal_supported_candidate';src=r/f'human_motion_data/{reference}';dst=r/f'human_motion_data/{candidate}';shutil.copytree(src,dst,dirs_exist_ok=True);backup=r/f'research/terminal-support-reference-{time.time_ns()}';backup.mkdir();path=next((src/'processed').rglob('motion.parquet'));audit=r/f'evaluation/{reference}-collision-audit.json';fidelity=r/'evaluation/physical-reference-source-fidelity.json';selection=r/'research/collision-reference-selection.json'
    if not fidelity.exists():subprocess.run([PY,str(BASE/'assess_physical_reference.py'),'--run',str(r),'--reference',reference],check=True)
    for f,name in [(path,'motion.parquet'),(audit,'audit.json'),(fidelity,'source-fidelity.json'),(selection,'selection.json')]:shutil.copy2(f,backup/name)
    before_source=json.loads(fidelity.read_text());before_audit=json.loads(audit.read_text());newton,retarget=containers(r,ep)
    for name in ['native_reference_audit','refine_native_object_poses','optimize_native_grasps','apply_collision_transitions','diagnose_terminal_support']:shutil.copy2(BASE/f'{name}.py',r/f'scripts/{name}.py')
    def native(script,label,**env):docker(['exec',newton,'env',*[f'{k}={v}' for k,v in env.items()],'python','-u',f'/episode/scripts/{script}.py'],r/f'logs/{label}.log')
    try:
     subprocess.run([PY,str(BASE/'apply_terminal_support.py'),'--run',str(r),'--reference',candidate],check=True)
     native('native_reference_audit','terminal-supported-collision-audit',V2D_AUDIT_REFERENCE=candidate);candidate_audit=r/f'evaluation/{candidate}-collision-audit.json';a=json.loads(candidate_audit.read_text())
     if any(any(hit['penetration_mm']>4 and not any('sharpa_wave' in n for n in hit['bodies']) for hit in f['penetrating_cross_body_contacts']) for f in a['frames']):
      native('refine_native_object_poses','terminal-supported-object-path',V2D_OBJECT_POSE_REFERENCE=candidate,V2D_PRESERVE_STATIC_POSES='1',V2D_OBJECT_POSE_MOBILE_OBJECTS=json.dumps([search['object_index']]),V2D_OBJECT_POSE_START_FRAME=str({14:80,16:98,20:132,28:146,30:212}[ep]),V2D_OBJECT_POSE_END_FRAME=str({14:107,16:127,20:158,28:189,30:239}[ep]),V2D_OBJECT_POSE_OUTPUT='research/terminal-supported-object-path.json')
      native('native_reference_audit','terminal-supported-collision-audit',V2D_AUDIT_REFERENCE=candidate);a=json.loads(candidate_audit.read_text())
     if any(any(hit['penetration_mm']>5 and any('sharpa_wave' in n for n in hit['bodies']) for hit in f['penetrating_cross_body_contacts']) for f in a['frames']):
      native('optimize_native_grasps','terminal-supported-hand-clearance',V2D_GRASP_REFERENCE=candidate,V2D_OPTIMIZE_CONTACT_FRAMES='1',V2D_CONTACT_FIT_OUTPUT='research/terminal-supported-hand-clearance.json')
      docker(['exec',retarget,'env',f'V2D_TRANSITION_REFERENCE={candidate}','V2D_CONTACT_FIT_OUTPUT=research/terminal-supported-hand-clearance.json','V2D_CONTACT_APPLY_OUTPUT=research/terminal-supported-hand-clearance-applied.json','/isaac-sim/python.sh','-u','/episode/scripts/apply_collision_transitions.py'],r/'logs/terminal-supported-hand-clearance-apply.log')
      native('native_reference_audit','terminal-supported-collision-audit',V2D_AUDIT_REFERENCE=candidate);a=json.loads(candidate_audit.read_text())
     subprocess.run([PY,str(BASE/'assess_physical_reference.py'),'--run',str(r),'--reference',candidate,'--overlay'],check=True);after_source=json.loads(fidelity.read_text());shutil.copy2(fidelity,backup/'candidate-source-fidelity.json')
     native('diagnose_terminal_support','terminal-supported-rest-test',V2D_DIAGNOSTIC_REFERENCE=candidate,V2D_DIAGNOSTIC_LABEL='terminal-supported-rest-test');rest=json.loads((r/'evaluation/terminal-supported-rest-test.json').read_text());obj=next(x for x in rest['objects'] if x['object']==search['object'])
     image_ok=all(after_source['aggregate'][n]['mean_iou']>=before_source['aggregate'][n]['mean_iou']-.08 for n in c['objects']);physics_ok=a['maximum_cross_body_penetration_mm']<=max(12.,before_audit['maximum_cross_body_penetration_mm']+2) and a['source_frames_over_10mm']<=before_audit['source_frames_over_10mm']+2;stable=rest['all_states_finite'] and obj['maximum_position_drift_m']<=.015 and obj['maximum_rotation_drift_rad']<=.2;accepted=image_ok and physics_ok and stable
     if accepted:
      atomic_copy(next((dst/'processed').rglob('motion.parquet')),path);a['reference']=reference;tmp=audit.with_suffix('.support.tmp');tmp.write_text(json.dumps(a,indent=2));tmp.replace(audit);after_source['reference']=reference;tmp=fidelity.with_suffix('.support.tmp');tmp.write_text(json.dumps(after_source,indent=2));tmp.replace(fidelity)
      value=json.loads((backup/'selection.json').read_text());value.update(contact_refinement_complete=True,physical_reference_source_fidelity=after_source['aggregate'],after_frames_over_10mm=a['source_frames_over_10mm'],final_maximum_penetration_mm=a['maximum_cross_body_penetration_mm']);selection.write_text(json.dumps(value,indent=2))
     else:atomic_copy(backup/'source-fidelity.json',fidelity);atomic_copy(backup/'selection.json',selection)
     report={'method':__doc__,'accepted':accepted,'source_image_guard_passed':image_ok,'collision_guard_passed':physics_ok,'terminal_gravity_test_passed':stable,'source_fidelity_before':before_source['aggregate'],'source_fidelity_after':after_source['aggregate'],'maximum_cross_body_penetration_mm':a['maximum_cross_body_penetration_mm'],'source_frames_over_10mm':a['source_frames_over_10mm'],'terminal_support_test':rest,'reference':reference,'candidate_reference':candidate,'backup':str(backup.relative_to(r)),'manual_visual_review_pending':True};(r/'research/terminal-support-reference-selection.json').write_text(json.dumps(report,indent=2));print(ep,'SUPPORTED REFERENCE',accepted,report,flush=True)
    except BaseException:
     atomic_copy(backup/'source-fidelity.json',fidelity);atomic_copy(backup/'selection.json',selection);raise
    break
  time.sleep(15)
