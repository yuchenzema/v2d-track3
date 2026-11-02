"""Validate the final brush learner against a complete preserved feature checkpoint."""
import json,shutil,time
from improvement_common import BASE,save,utc,digest
from batch_native import _docker


def main():
    old=BASE/'runs/episode_000030_newton_stable_brush_20261021'
    root=BASE/'runs/episode_000030_newton_consistent_brush_20261021'
    step=min(int(p.stem.split('_')[-1]) for p in (old/'flash_chord_coherent_actor_transfer').glob('state_*.safetensors') if int(p.stem.split('_')[-1])>=35_000_000)
    assert not (root/'coherent-input-checksums.json').exists()
    identifiers=[]
    for kind in ['policy','state']:
        source=old/f'flash_chord_coherent_actor_transfer/{kind}_{step}.safetensors'
        assert source.exists() and time.time()-source.stat().st_mtime>5
        with source.open('rb') as stream:
            size=int.from_bytes(stream.read(8),'little');header=json.loads(stream.read(size))
        assert source.stat().st_size==8+size+max(v['data_offsets'][1] for k,v in header.items() if k!='__metadata__')
        identifiers.append(header['__metadata__']['user.checkpoint_set_id'])
        target=root/'research/initial-policy'/source.name;shutil.copy2(source,target);assert digest(source)==digest(target)
    assert len(set(identifiers))==1
    cfg=json.loads((root/'episode-config.json').read_text());remaining=45_001_728-step
    cfg.update(training_initial_counter=step,training_budget=remaining,contact_training_additional_budget=remaining,
        reconstruction_review_notes=cfg.get('reconstruction_review_notes','')+' Source recheck20261021: stable holder rest; lift before lateral extraction; consistent brush yaw branch removes an unobserved179.6-degree wrist turn. Corrected extraction turns29.6 degrees.')
    save(root/'episode-config.json',cfg)
    for name in ['consistent-brush-grip-reference.json','clearance-brush-pickup-reference.json','stable-brush-pickup-reference.json']:
        path=old/'research'/name
        if path.exists():shutil.copy2(path,root/'research'/name)
    for name in ['consistent-brush-initial-rest.json','consistent-brush-source-fidelity.json','track3_consistent_brush_pickup_candidate-collision-audit.json']:
        shutil.copy2(old/'evaluation'/name,root/'evaluation'/name)
    selection=json.loads((root/'research/physical-grasp-reference-selection.json').read_text())
    selection.update(reference='track3_consistent_brush_pickup_candidate',physical_reference_source_report='evaluation/consistent-brush-source-fidelity.json',initial_support_evidence='evaluation/consistent-brush-initial-rest.json',
        approved_for_independent_training=True,verified_policy_improvement=False,
        remaining_problem='Learn continuous dynamic grip with the corrected extraction and yaw branch. Untrained control still slips after source60; no learned improvement claimed before independent evaluation.')
    save(root/'research/physical-grasp-reference-selection.json',selection)
    for source,target in [('contact_finetune_training.py','train_coherent_actor_transfer.py'),('freeze_actor_normalization.py','freeze_actor_normalization.py')]:shutil.copy2(BASE/source,root/'scripts'/target)
    for option,label in [('V2D_VALIDATE_CONFIG_ONLY=1','config'),('V2D_VALIDATE_ACTOR_TRANSFER_ONLY=1','zero-mean-transfer')]:
        _docker(['exec',cfg['native_container'],'env','CUDA_VISIBLE_DEVICES=','JAX_PLATFORMS=cpu',option,'python','-u','/episode/scripts/run_geometry_cpu.py','/episode/scripts/train_coherent_actor_transfer.py'],root/f'logs/final-consistent-{label}-validation.log')
    save(root/'research/preparation-status.json',{'updated_utc':utc(),'status':'prepared_and_configuration_validated','source_run':str(old),'seed_checkpoint_environment_steps':step,
        'additional_training_budget':remaining,'pretrained_features_transferred':True,'residual_head_reinitialized_zero':True,
        'fresh_critic_optimizer_normalizer_and_replay':True,'initial_rest_verified':True,'full_unassisted_learned_rollout_not_yet_verified':True})
    print('Final consistent brush training configuration validated; remaining new steps:',remaining,flush=True)


if __name__=='__main__':main()
