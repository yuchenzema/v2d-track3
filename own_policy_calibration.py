"""Declared inference calibration of a learned own-geometry policy.

Weights and actual training counters are preserved; the original training
configuration is retained separately. Evaluation uses the unchanged official
entry point, the original reconstructed objects and original45N/Nm limits.
"""
import hashlib,json
from pathlib import Path
import numpy as np
from safetensors import safe_open
from safetensors.numpy import load_file,save_file
import pyarrow.parquet as pq
from improvement_common import digest,save,utc


def weights_digest(tensors):
    h=hashlib.sha256()
    for name,value in sorted(tensors.items()):
        h.update(name.encode());h.update(value.dtype.str.encode());h.update(str(value.shape).encode());h.update(value.tobytes())
    return h.hexdigest()


def derive(root,parent,profile):
    from own_geometry import validate
    root,parent=Path(root),Path(parent);validate(root)
    with safe_open(parent,framework='numpy') as source:metadata=source.metadata()
    cfg=json.loads(metadata['user.resolved_config_json']);assert float(cfg['env']['voc_scale'])==0.
    kp,kd,angular=profile['kp'],profile['kd'],profile['angular_kp']
    assert 25<=kp<=500 and 6<=kd<=40 and 25<=angular<=250
    original=metadata['user.resolved_config_json']
    overrides=cfg['embodiment']['config'].get('joint_overrides') or {
        r'wrist_[xyz]$':dict(kp=25.,kd=6.,effort_limit=45.),
        r'right_wrist_y':dict(default_pos=.25),r'left_wrist_y':dict(default_pos=-.25),
        r'wrist_z$':dict(default_pos=.5),r'wrist_rotvec$':dict(mode='effort',kp=25.,kd=6.,effort_limit=45.)}
    overrides=dict(overrides);overrides[r'wrist_[xyz]$']={**overrides[r'wrist_[xyz]$'],**dict(kp=kp,kd=kd,effort_limit=45.)}
    cfg['embodiment']['config']['joint_overrides']=overrides
    assert cfg['action']['wrist_tau_max']==45.
    cfg['action']['wrist_kp']=angular
    tensors=load_file(parent);weight_sha=weights_digest(tensors)
    metadata.update({'user.original_training_resolved_config_json':original,
        'user.resolved_config_json':json.dumps(cfg,sort_keys=True),
        'user.control_calibration_parent_sha256':digest(parent),
        'user.control_calibration_weights_sha256':weight_sha,
        'user.control_calibration_profile_json':json.dumps(profile,sort_keys=True),
        'user.control_calibration_new_training_steps':'0'})
    tag=hashlib.sha256(json.dumps(profile,sort_keys=True).encode()).hexdigest()[:12]
    path=root/f"quality_calibration/{tag}/policy_{metadata['user.environment_steps']}.safetensors";path.parent.mkdir(parents=True,exist_ok=True)
    if not path.exists():save_file(tensors,str(path),metadata=metadata)
    assert weights_digest(load_file(path))==weight_sha
    save(path.with_suffix('.derivation.json'),dict(updated_utc=utc(),parent=str(parent),parent_sha256=digest(parent),
        checkpoint=str(path),checkpoint_sha256=digest(path),weights_sha256=weight_sha,
        actual_parent_training_steps=int(metadata['user.environment_steps']),additional_training_steps=0,
        original_training_configuration_preserved=True,inference_control_profile=profile,actuator_limits_preserved=True))
    return path


def diagnose(trajectory):
    table=pq.read_table(trajectory).to_pandas();names=sorted(table['object_id'].unique());errors=[];actual=[];reference=[]
    for name in names:
        rows=table[table.object_id==name].sort_values('step')
        a=rows[[f'achieved_position_{axis}' for axis in 'xyz']].to_numpy();r=rows[[f'reference_position_{axis}' for axis in 'xyz']].to_numpy()
        actual.append(a);reference.append(r);errors.extend(np.linalg.norm(a-r,axis=1)*100)
    pair=[]
    for i in range(len(names)):
        for j in range(i):pair.extend(np.linalg.norm((actual[i]-actual[j])-(reference[i]-reference[j]),axis=1)*100)
    return dict(mean_source_error_cm=float(np.mean(errors)),p95_source_error_cm=float(np.quantile(errors,.95)),
        pair_source_error_cm=float(np.mean(pair)) if pair else 0.,
        dropped_frames=sum(int(np.sum(a[:,2]<r[:,2]-.12)) for a,r in zip(actual,reference)),
        local_source_proxy_not_kaggle_score=True)


def evaluate(root,checkpoint,container):
    from official_policy_suite import evaluate_candidate
    root=Path(root);suite=evaluate_candidate(root,checkpoint,container)
    trajectory=Path(suite['recorded_trajectory']);states=Path(suite['recorded_body_states']);sha=digest(checkpoint)
    path=root/f'research/control-calibration/{sha[:16]}.json'
    if path.is_file():
        old=json.loads(path.read_text())
        if old.get('trajectory_sha256')==digest(trajectory) and old.get('states_sha256')==digest(states):return path,old
    result=dict(status='ready_for_submission',episode=suite['episode'],run=str(root),updated_utc=utc(),
        checkpoint=str(checkpoint),checkpoint_sha256=sha,actual_training_steps=suite['attempts']['recorded']['actual_training_steps'],
        trajectory=str(trajectory),trajectory_sha256=digest(trajectory),states_sha256=digest(states),
        frozen_input_manifest_sha256=digest(root/'coherent-input-checksums.json'),public_object_scans_used=False,
        additional_training_steps=0,strict_completion_granted=False,control_calibration=True,
        quality_diagnostic=diagnose(trajectory))
    save(path,result)
    return path,result
