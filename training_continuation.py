"""Extend one validated experiment without changing frozen geometry or targets."""
import json
from pathlib import Path
from improvement_common import digest,save,utc


def extend(root,increment=12_800_000):
    root=Path(root);cfg=json.loads((root/'episode-config.json').read_text())
    if cfg.get('experimental_training'):
        save(root/'research/experimental-continuation-skipped.json',dict(updated_utc=utc(),
             reason='Experimental pilot budget remains capped; a failed final audit does not authorize another 12.8M steps',
             training_budget=cfg['training_budget']))
        return cfg['training_budget']
    assert cfg.get('validated_video_pipeline') and cfg.get('training_curriculum')=='fresh_rebuild'
    for name,expected in json.loads((root/'coherent-input-checksums.json').read_text()).items():
        assert digest(root/name)==expected,('Frozen reference changed; fresh reconstruction validation required',name)
    folder=root/'flash_chord_coherent_actor_transfer'
    states=[p for p in folder.glob('state_*.safetensors') if p.with_name(p.name.replace('state_','policy_')).is_file()]
    assert states,'No resumable actor and learner checkpoint pair'
    latest=max(states,key=lambda p:int(p.stem.split('_')[-1]));step=int(latest.stem.split('_')[-1])
    cfg.update(training_budget=step+increment,training_continuation_round=cfg.get('training_continuation_round',0)+1)
    save(root/'episode-config.json',cfg)
    save(root/f"research/training-continuation-{cfg['training_continuation_round']:03d}.json",
         dict(updated_utc=utc(),from_checkpoint=str(latest),from_checkpoint_sha256=digest(latest),
              resume_environment_steps=step,additional_environment_steps_requested=increment,
              target_environment_steps=cfg['training_budget'],frozen_input_hashes_verified=True,
              optimizer_restore_required=True,object_assistance_force_N=0.,object_assistance_torque_Nm=0.))
    (folder/'TRAINING_COMPLETED').unlink(missing_ok=True)
    return cfg['training_budget']
