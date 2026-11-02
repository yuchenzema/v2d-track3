"""Choose the smallest tested integration step that preserves observed passive object rests."""
import fcntl,json,shutil,subprocess,sys,time
from batch_worker import BASE,PY,context
from batch_native import containers,docker

def severity(report):return max([max(x['maximum_position_drift_m']/.01,x['maximum_orientation_drift_rad']/.2) for x in report['objects']],default=0.)

for ep in map(int,sys.argv[1:]):
 r,c=context(ep);baseline=r/'evaluation/passive-rest-preflight-initial-rest.json'
 while not baseline.exists():time.sleep(10)
 with (r/'research/collision-reference.lock').open('a') as lock:
  fcntl.flock(lock,fcntl.LOCK_EX);assert not (r/'coherent-input-checksums.json').exists()
  c=json.loads((r/'episode-config.json').read_text());old_steps=c.get('physics_substeps',5);old=json.loads(baseline.read_text());rows=[{'substeps':old_steps,'label':'passive-rest-preflight','report':old}]
  if severity(old)>1:
   newton,_=containers(r,ep);shutil.copy2(BASE/'diagnose_zero_residual.py',r/'scripts/diagnose_zero_residual.py')
   for substeps in [20,40]:
    if substeps<=old_steps:continue
    label=f'passive-rest-{20*substeps}hz-diagnostic';path=r/f'evaluation/{label}-initial-rest.json'
    if not path.exists():docker(['exec',newton,'env',f'V2D_DIAGNOSTIC_SUBSTEPS={substeps}',f'V2D_DIAGNOSTIC_LABEL={label}','python','-u','/episode/scripts/diagnose_zero_residual.py'],r/f'logs/{label}.log')
    subprocess.run([PY,str(BASE/'assess_passive_rests.py'),'--run',str(r),'--label',label],check=True);report=json.loads(path.read_text());rows.append({'substeps':substeps,'label':label,'report':report})
    if severity(report)<=1:break
  passing=[x for x in rows if severity(x['report'])<=1];selected=min(passing,key=lambda x:x['substeps']) if passing else min(rows,key=lambda x:severity(x['report']));changed=selected['substeps']!=old_steps and severity(selected['report'])<severity(old)*.8
  if changed:
   old_config=r/'research/config-before-passive-rest-integration.json'
   if not old_config.exists():shutil.copy2(r/'episode-config.json',old_config)
   c.update(physics_substeps=selected['substeps'],physics_integration_hz=20*selected['substeps'],physics_integration_reason='Select smallest tested timestep with <=1 cm translation and <=0.2 rad rotation before actual first hand contact. Unassisted zero-residual diagnostic only; source timeline, geometry and policy control rate unchanged.',passive_rest_diagnostic=f"evaluation/{selected['label']}-initial-rest.json");temporary=r/'episode-config.json.tmp';temporary.write_text(json.dumps(c,indent=2));temporary.replace(r/'episode-config.json')
  decision={'method':__doc__,'configuration_changed':changed,'previous_physics_hz':20*old_steps,'selected_physics_hz':20*selected['substeps'],'selected_passes_passive_rest_check':severity(selected['report'])<=1,'trials':[{'substeps':x['substeps'],'diagnostic':f"evaluation/{x['label']}-initial-rest.json",'maximum_normalized_rest_drift':severity(x['report'])} for x in rows]};(r/'research/passive-rest-integration-selection.json').write_text(json.dumps(decision,indent=2));print(ep,decision,flush=True)
