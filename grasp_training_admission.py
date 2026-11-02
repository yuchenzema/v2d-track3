"""Keep physical probes, but allow explicitly authorized experimental learning."""


def decide(cfg,policy,probe,reference_sha256):
    assert probe.get('reference_parquet_sha256')==reference_sha256,'Grasp probe belongs to another reference'
    assert probe.get('diagnostic_only') is True and probe.get('policy_inference') is False,'Invalid pretraining probe provenance'
    assert probe.get('source_frame_playback') is True,'Probe source timing missing'
    assert probe.get('object_force_assistance_N')==0. and probe.get('object_torque_assistance_Nm')==0.,'Probe used object assistance'
    assert probe.get('grasps'),'No grasp observations'
    passed=probe.get('all_grasps_passed') is True
    learn_from_failed_grasp=(cfg.get('experimental_training') is True
        and policy.get('enabled') is True and cfg.get('episode') in policy.get('episodes',[])
        and policy.get('require_successful_pretraining_grasp') is False)
    assert passed or learn_from_failed_grasp,'Physical approach/hold/lift test failed; successful pretraining grasp is required by policy'
    return dict(training_admitted=True,pretraining_grasp_passed=passed,
        pretraining_grasp_is_diagnostic=learn_from_failed_grasp,
        post_training_grasp_acceptance_required=True,reference_parquet_sha256=reference_sha256,
        zero_object_assistance_required=True,nonpenetrating_reference_required=True,
        experimental_only=bool(cfg.get('experimental_training')),strict_completion_granted=False)
