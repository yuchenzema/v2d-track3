"""Fit deep hand collisions throughout a fresh reference, then test unassisted grasps."""
import argparse
import json
import shutil
import subprocess
from pathlib import Path
import numpy as np
import pyarrow.parquet as pq
from improvement_common import BASE, save, utc, digest
from batch_native import _docker, docker

p=argparse.ArgumentParser();p.add_argument('--run',type=Path,required=True);r=p.parse_args().run
c=json.loads((r/'episode-config.json').read_text());assert not (r/'coherent-input-checksums.json').exists()
marker=r/'research/full-trajectory-contact-fit.json'
strict_contacts=c.get('require_nonpenetrating_reference_before_training',False)
if strict_contacts:marker=r/'research/full-trajectory-contact-fit-v2-nonpenetration.json'
def repair_physical_holds():
    if c['episode']==20 and c.get('require_physical_holds_before_training'):
        subprocess.run([str(r/'.venv-video/bin/python'),str(BASE/'repair_episode20_physical_holds_20261027.py'),'--run',str(r)],check=True)
def require_clean_reference():
    if not strict_contacts:return
    cpu('native_reference_audit.py','nonpenetration-before-training',V2D_AUDIT_REFERENCE=reference,V2D_COLLISION_EPSILON_M='0.000001')
    audit=json.loads((r/f'evaluation/{reference}-collision-audit.json').read_text())
    passed=audit['maximum_cross_body_penetration_mm']<=.001
    save(r/'research/reference-nonpenetration-gate.json',{'updated_utc':utc(),'passed':passed,
         'all_source_frames_checked':c['frames'],'maximum_penetration_mm':audit['maximum_cross_body_penetration_mm'],
         'reference_parquet_sha256':digest(next((r/f'human_motion_data/{reference}/processed').rglob('motion.parquet'))),
         'numerical_epsilon_m':1e-6,'training_allowed':passed})
    assert passed,'Reconstructed robot/object/table poses still intersect; policy training is blocked'
if marker.exists():
    native=c['native_container'];reference='track3_collision_variant'
    def cpu(script,label,**env):
        _docker(['exec',native,'env','CUDA_VISIBLE_DEVICES=','JAX_PLATFORMS=cpu',*[f'{k}={v}' for k,v in env.items()],
                 'python','-u','/episode/scripts/run_geometry_cpu.py','/episode/scripts/'+script],r/f'logs/{label}.log')
    repair_physical_holds();require_clean_reference();raise SystemExit(0)
native=c['native_container'];retarget=native.removesuffix('newton')+'retarget'
reference='track3_collision_variant';path=next((r/f'human_motion_data/{reference}/processed').rglob('motion.parquet'))
original=pq.ParquetFile(path).read(columns=['object_body_position','object_body_wxyz']).to_pylist()
shutil.copy2(path,r/'research/motion-before-full-trajectory-fit.parquet')
def cpu(script,label,**env):
    _docker(['exec',native,'env','CUDA_VISIBLE_DEVICES=','JAX_PLATFORMS=cpu',*[f'{k}={v}' for k,v in env.items()],
             'python','-u','/episode/scripts/run_geometry_cpu.py','/episode/scripts/'+script],r/f'logs/{label}.log')
rounds=[]
for iteration in range(4 if strict_contacts else 2):
    cpu('native_reference_audit.py',f'full-contact-audit-{iteration}',V2D_AUDIT_REFERENCE=reference,
        V2D_COLLISION_EPSILON_M='0.000001' if strict_contacts else '0.001')
    audit=json.loads((r/f'evaluation/{reference}-collision-audit.json').read_text())
    selected={side:[row['source_frame'] for row in audit['frames'] if any(x['penetration_mm']>(.001 if strict_contacts else 3.) and any(n.startswith(side+'_sharpa_wave_') for n in x['bodies']) for x in row['penetrating_cross_body_contacts'])] for side in ['left','right']}
    if not any(selected.values()):break
    fit=f'research/full-contact-fit-{iteration}.json'
    cpu('optimize_native_grasps.py',f'full-contact-fit-{iteration}',V2D_GRASP_REFERENCE=reference,
        V2D_OPTIMIZE_CONTACT_FRAMES='1',V2D_SOURCE_HAND_SIGMA='24',V2D_CONTACT_FRAME_SELECTION=json.dumps(selected),V2D_CONTACT_FIT_OUTPUT=fit)
    docker(['exec',retarget,'env',f'V2D_TRANSITION_REFERENCE={reference}',f'V2D_CONTACT_FIT_OUTPUT={fit}',
            f'V2D_CONTACT_APPLY_OUTPUT=research/full-contact-applied-{iteration}.json','V2D_BLEND_ADJACENT_FITS=1',
            '/isaac-sim/python.sh','-u','/episode/scripts/apply_collision_transitions.py'],r/f'logs/full-contact-apply-{iteration}.log')
    assert original==pq.ParquetFile(path).read(columns=['object_body_position','object_body_wxyz']).to_pylist(),'Object trajectory changed during hand fitting'
    rounds.append({'iteration':iteration,'source_frames_with_deep_hand_contacts':selected,'fit':fit})
cpu('native_reference_audit.py','full-contact-final-audit',V2D_AUDIT_REFERENCE=reference)
text=(BASE/'diagnose_static_grasp.py').read_text().replace("'sim.njmax=1536'","'sim.njmax=4096','sim.nconmax=2048'")
(r/'scripts/diagnose_static_grasp.py').write_text(text)
frame=min(c['frames']-1,max(g['start'] for g in c['grasps'])+12)
docker(['exec',native,'env',f'V2D_STATIC_GRASP_SOURCE_FRAME={frame}',
        'V2D_WRIST_MOTOR_PROFILE='+json.dumps(c['wrist_motor_profile']),
        'python','-u','/episode/scripts/diagnose_static_grasp.py'],r/'logs/rebuilt-static-grasp-test.log')
probe=r/f'evaluation/static-grasp-hold-lift-source-{frame}-v2-kp250-kd20.json'
result=json.loads(probe.read_text());audit=json.loads((r/f'evaluation/{reference}-collision-audit.json').read_text())
fidelity=json.loads((r/'evaluation/reference-source-fidelity.json').read_text())
assert [frame['frame'] for frame in fidelity['frames']]==list(range(c['frames']))
c['all_source_frames_decoded_and_automatically_checked']=True
save(r/'episode-config.json',c)
fit_report={'updated_utc':utc(),'rounds':rounds,'all_source_frames_collision_checked':c['frames'],
             'object_trajectories_preserved':True,'static_probe_is_learned_policy':False,'unassisted_grasp_probe':result,
             'final_maximum_cross_body_penetration_mm':audit['maximum_cross_body_penetration_mm'],
             'grasp_stability_issues_remain':any(not x.get('stable_hold_and_lift',x.get('passive_rest_stable',False)) for x in result['objects'].values())}
repair_physical_holds()
require_clean_reference()
save(marker,fit_report)
