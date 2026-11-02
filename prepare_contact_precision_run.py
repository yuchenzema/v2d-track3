"""Prepare a separate, validated contact precision continuation for episode30/32."""
import argparse,json,shutil,stat
from pathlib import Path
from improvement_common import BASE,save,utc,digest
from batch_native import _docker


def prepare(ep,source,step,reference,variant,budget,focus,full_state=False,reset_head=False):
    source=Path(source);root=BASE/'runs'/f'episode_{ep:06d}_newton_{variant}'
    assert ep in (30,32) and not root.exists()
    cfg=json.loads((source/'episode-config.json').read_text())
    assert cfg['episode']==ep and (source/'human_motion_data'/reference).is_dir()
    root.mkdir()
    for name in ['logs','research','evaluation','videos','deliverables','flash_chord_coherent_actor_transfer']:
        (root/name).mkdir()
    for name in ['reconstruction','scripts']:
        shutil.copytree(source/name,root/name)
    for name in set([reference,'track3_video']):
        shutil.copytree(source/'human_motion_data'/name,root/'human_motion_data'/name)
    (root/'.venv-video').symlink_to(BASE/'runs/episode_000005_newton_20261015/.venv-video',target_is_directory=True)
    cfg.update(run=str(root),training_variant=variant,native_container=f'v2d-precisionfull-episode{ep}-newton' if full_state else f'v2d-precision-episode{ep}-newton',
               training_curriculum='contact_finetune',training_reference=reference,
               training_initial_counter=step,training_world_count=int(cfg.get('training_world_count',1024)) if full_state else 512,training_budget=budget,
               contact_training_additional_budget=budget,training_compute_dtype='float32',
               training_numeric_failure_guard=True,physics_substeps=20,physics_integration_hz=400,
               physics_nconmax=2048 if ep==30 else 512,physics_njmax=4096 if ep==30 else 1536,
               improvement_experiment='Video-reviewed reconstruction and centimetre-scale contact precision finetuning')
    for key in ['adaptive_single_prefix','backward_hold_source_frame','strict_reset_retrain_prefix_steps','training_seed_checkpoint_run','contact_full_state_seed']:
        cfg.pop(key,None)
    if variant=='conservative_spoon_20261021':cfg['native_container']=f'v2d-conservative-episode{ep}-newton'
    if variant.startswith(('regularized_','consistent_')):cfg['native_container']=f'v2d-{variant.split("_")[0]}-episode{ep}-newton'
    if full_state:cfg.update(contact_full_state_seed=True,training_seed_checkpoint_run=str(source))
    if reset_head:
        assert not full_state
        cfg.update(reset_residual_head_for_new_reference=True,training_actor_learning_rate_multiplier=.1,
                   training_behavior_cloning_coefficient=1.,contact_replay_capacity=524288,contact_replay_minimum_size=262144,
                   contact_checkpoint_interval=2_000_000)
    save(root/'episode-config.json',cfg)
    seed=root/'research/initial-policy';seed.mkdir();hashes={}
    for kind in ['policy','state']:
        path=source/f'flash_chord_coherent_actor_transfer/{kind}_{step}.safetensors'
        if not path.exists():path=source/f'research/initial-policy/{kind}_{step}.safetensors'
        target=seed/path.name;shutil.copy2(path,target);hashes[path.name]=digest(path);assert digest(target)==hashes[path.name]
    for name in ['source-video-review.json','physical-grasp-reference-selection.json']:
        if (source/'research'/name).exists():shutil.copy2(source/'research'/name,root/'research'/name)
    selection=json.loads((root/'research/physical-grasp-reference-selection.json').read_text())
    selection.update(reference=reference,updated_utc=utc(),training_curriculum='contact_finetune',
                     verified_policy_improvement=False,training_initial_counter=step)
    save(root/'research/physical-grasp-reference-selection.json',selection)
    for name in ['reference-source-fidelity.json','physical-reference-source-fidelity.json']:
        if (source/'evaluation'/name).exists():shutil.copy2(source/'evaluation'/name,root/'evaluation'/name)
    save(root/'research/contact-finetune-adjustments.json',{'updated_utc':utc(),'source_interval':focus,
         'residual_mean_learning_rate_multiplier':.1,'training_translation_precision':{'fraction':.1 if full_state else .25,'scale':.02},
         'reason':'Improve physical object placement at the observed source times; retain native orientation reward and independent unmodified evaluation.'})
    for helper in ['contact_finetune_training.py','training_finger_reset.py','translation_precision_reward.py','training_numeric_failure_guard.py','validation_stop.py','freeze_actor_normalization.py']:
        shutil.copy2(BASE/helper,root/'scripts'/('train_coherent_actor_transfer.py' if helper=='contact_finetune_training.py' else helper))
    info=json.loads(_docker(['inspect',json.loads((source/'episode-config.json').read_text())['native_container']]).stdout)[0]
    args=['run','-d','--gpus','all','--runtime','nvidia','--ipc','host','--name',cfg['native_container']]
    for mount in info['Mounts']:
        origin=str(root) if mount['Destination']=='/episode' else mount['Source']
        args+=['-v',origin+':'+mount['Destination']+('' if mount['RW'] else ':ro')]
    for node in sorted([*Path('/dev').glob('nvidia*'),*Path('/dev/nvidia-caps').glob('*')]):
        if stat.S_ISCHR(node.stat().st_mode):args+=['--device',str(node)]
    args+=['-e','PYGLET_HEADLESS=true','-e','XLA_PYTHON_CLIENT_PREALLOCATE=false','--workdir','/workspace','--entrypoint','/bin/sleep',info['Image'],'infinity']
    _docker(args,root/'logs/create-newton.log')
    validations=[('V2D_VALIDATE_CONFIG_ONLY=1','configuration')]
    if not full_state:validations.append(('V2D_VALIDATE_ACTOR_TRANSFER_ONLY=1','actor-transfer'))
    for option,label in validations:
        _docker(['exec',cfg['native_container'],'env','CUDA_VISIBLE_DEVICES=','JAX_PLATFORMS=cpu',option,
                 'python','-u','/episode/scripts/run_geometry_cpu.py','/episode/scripts/train_coherent_actor_transfer.py'],root/f'logs/precision-{label}-validation.log')
    save(root/'research/preparation-status.json',{'updated_utc':utc(),'status':'prepared_and_configuration_validated',
         'source_run':str(source),'seed_checkpoint_environment_steps':step,'seed_checkpoint_sha256':hashes,
         'additional_training_budget':budget,'actor_only_transfer':not full_state,'object_assistance':0.,'training_started':False})
    print('Prepared:',root,flush=True)
    return root


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('episode',type=int);p.add_argument('source',type=Path);p.add_argument('step',type=int)
    p.add_argument('reference');p.add_argument('variant');p.add_argument('--budget',type=int,default=20_000_768);p.add_argument('--focus',nargs=2,type=int,default=[36,170]);p.add_argument('--full-state',action='store_true');p.add_argument('--reset-residual-head',action='store_true');a=p.parse_args()
    prepare(a.episode,a.source,a.step,a.reference,a.variant,a.budget,a.focus,a.full_state,a.reset_residual_head)
