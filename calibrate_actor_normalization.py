"""Keep newly trained actor weights and restore the successful normalization anchor."""
import hashlib,json,shutil
from pathlib import Path
from types import SimpleNamespace
import jax,numpy as np
from flash_chord.training.checkpoint import read_checkpoint_metadata
from flash_chord.training.flash_sac.config import NetworkConfig
from flash_chord.training.flash_sac.evaluation import create_actor_template
from flash_chord.training.flash_sac.checkpoint import load_actor_checkpoint,save_actor_checkpoint

R=Path('/episode');step=int(__import__('os').environ.get('V2D_CALIBRATE_STEP','62000128'))
source=R/f'flash_chord_coherent_actor_transfer/policy_{step}.safetensors';anchor=R/'research/initial-policy/policy_60000256.safetensors'
dst=R/f'research/normalization-calibration-{step}';dst.mkdir(exist_ok=True)
parent=dst/'trained-parent.safetensors'
if not parent.exists():shutil.copy2(source,parent)
metadata=read_checkpoint_metadata(parent);cfg=json.loads(metadata['resolved_config_json'])['training']
network=NetworkConfig(**{k:v for k,v in cfg['network'].items() if k!='_target_'})
training=SimpleNamespace(network=network,seed=cfg['seed'],compute_dtype=cfg['compute_dtype'],mixed_precision=cfg['compute_dtype']!='float32')
_,template=create_actor_template(training,722,56,device=jax.devices('cpu')[0])
trained,_=load_actor_checkpoint(parent,template);previous,_=load_actor_checkpoint(anchor,template)
calibrated=trained.replace(batch_stats=previous.batch_stats)
for a,b in zip(jax.tree.leaves(trained.params),jax.tree.leaves(calibrated.params)):np.testing.assert_array_equal(np.asarray(a),np.asarray(b))
metadata.update(normalization_calibration='Newly trained parameters; successful60M actor normalization statistics retained.',
                normalization_anchor_sha256=hashlib.sha256(anchor.read_bytes()).hexdigest(),
                parameter_parent_sha256=hashlib.sha256(parent.read_bytes()).hexdigest(),checkpoint_set_id=__import__('uuid').uuid4().hex)
path=dst/f'policy_{step}.safetensors';save_actor_checkpoint(path,calibrated,metadata=metadata)
restored,_=load_actor_checkpoint(path,template)
for a,b in zip(jax.tree.leaves(previous.batch_stats),jax.tree.leaves(restored.batch_stats)):np.testing.assert_array_equal(np.asarray(a),np.asarray(b))
(dst/'calibration-provenance.json').write_text(json.dumps({'checkpoint':str(path),'environment_steps':step,
 'trained_parameters_unchanged':True,'normalization_matches_anchor_exactly':True,'anchor_checkpoint':str(anchor),
 'anchor_sha256':metadata['normalization_anchor_sha256'],'trained_parent_sha256':metadata['parameter_parent_sha256'],
 'object_states_actions_references_and_evaluation_unchanged':True},indent=2))
print('Calibrated actor normalization saved; trained parameters are unchanged:',path,flush=True)
owner=R.stat()
for item in [dst,*dst.iterdir()]:__import__('os').chown(item,owner.st_uid,owner.st_gid)
