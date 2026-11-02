"""Check new reconstruction support before committing a fresh training budget."""
import argparse,json,shutil
from pathlib import Path
from improvement_common import BASE,save,utc
from batch_native import _docker,docker
import subprocess
p=argparse.ArgumentParser();p.add_argument('--run',type=Path,required=True);a=p.parse_args();r=a.run;c=json.loads((r/'episode-config.json').read_text());marker=r/'research/rebuild-physical-preflight.json'
if marker.exists() and json.loads(marker.read_text()).get('passed'):raise SystemExit(0)
assert not (r/'coherent-input-checksums.json').exists(),'Never change a frozen new learner reference'
native=c['native_container']
for helper in ['native_reference_audit.py','project_native_object_contacts.py','diagnose_zero_residual.py']:
 text=(BASE/helper).read_text().replace("'sim.njmax=1536'","'sim.njmax=4096'").replace("'scene.world_count=1'","'scene.world_count=1','sim.nconmax=2048'")
 if helper=='diagnose_zero_residual.py':
  text=text.replace(" cfg=OmegaConf.create(apply_evaluation_composition", " from omegaconf import open_dict\n from flash_chord.embodiments.sharpa_hands import SharpaHandsConfig\n overrides=dict(SharpaHandsConfig().joint_overrides);overrides[r'wrist_[xyz]$']={**overrides[r'wrist_[xyz]$'],**c['wrist_motor_profile']}\n with open_dict(cfg.embodiment.config):cfg.embodiment.config.joint_overrides=overrides\n cfg=OmegaConf.create(apply_evaluation_composition")
 (r/'scripts'/helper).write_text(text)
def cpu(script,label,**env):_docker(['exec',native,'env','CUDA_VISIBLE_DEVICES=','JAX_PLATFORMS=cpu',*[f'{k}={v}' for k,v in env.items()],'python','-u','/episode/scripts/run_geometry_cpu.py','/episode/scripts/'+script],r/f'logs/{label}.log')
if not (r/'research/rebuild-object-projection-completed.json').exists():
 cpu('native_reference_audit.py','rebuild-object-hull-audit',V2D_AUDIT_REFERENCE='track3_coherent_variant')
 subprocess.run([str(r/'.venv-video/bin/python'),str(BASE/'project_collision_floor.py'),'--run',str(r)],check=True)
 cpu('project_native_object_contacts.py','rebuild-native-contact-projection',V2D_PROJECT_REFERENCE='track3_collision_variant',V2D_PROJECTION_BOUND_M='.025',V2D_PROJECTION_CAMERA_PENALTY='100')
 save(r/'research/rebuild-object-projection-completed.json',{'updated_utc':utc(),'raw_image_reconstruction_unchanged':True,'native_reference_projection_bounded_m':.025})
label='rebuild-passive-rest'
for attempt in range(3):
 docker(['exec',native,'env','V2D_DIAGNOSTIC_REFERENCE=track3_collision_variant','V2D_DIAGNOSTIC_SUBSTEPS=20',f'V2D_DIAGNOSTIC_LABEL={label}','python','-u','/episode/scripts/diagnose_zero_residual.py'],r/f'logs/{label}.log')
 report=json.loads((r/f'evaluation/{label}-initial-rest.json').read_text());passed=all(x['passed_1cm_0_2rad_passive_rest'] for x in report['objects']);save(marker,{'updated_utc':utc(),'passed':passed,'unassisted_passive_rest':report,'no_learned_policy_used':True,'training_started':False})
 if passed:break
 if attempt==2:raise AssertionError('New reconstructed support remains unstable; repair before training')
 subprocess.run([str(r/'.venv-video/bin/python'),str(BASE/'refine_initial_rest_20261022.py'),'--run',str(r)],check=True)
 changes=json.loads((r/'research/rebuilt-initial-support-refinement.json').read_text())['attempts'][-1]['changes']
 assert changes,'No source-consistent bounded support repair was found; keep the reconstruction under review'
print('Fresh reconstruction passive support verified',c['episode'],flush=True)
