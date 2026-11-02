import unittest
from grasp_training_admission import decide


class GraspAdmissionTests(unittest.TestCase):
    def test_pilot_does_not_automatically_extend_budget_or_remove_completion_marker(self):
        import json,tempfile
        from pathlib import Path
        from training_continuation import extend
        with tempfile.TemporaryDirectory() as directory:
            root=Path(directory);(root/'flash_chord_coherent_actor_transfer').mkdir()
            marker=root/'flash_chord_coherent_actor_transfer/TRAINING_COMPLETED';marker.write_text('done')
            (root/'episode-config.json').write_text(json.dumps(dict(experimental_training=True,training_budget=1280000)))
            self.assertEqual(extend(root),1280000)
            self.assertTrue(marker.exists())
            self.assertEqual(json.loads((root/'episode-config.json').read_text())['training_budget'],1280000)

    def setUp(self):
        self.cfg=dict(episode=38,experimental_training=True)
        self.policy=dict(enabled=True,episodes=[38],require_successful_pretraining_grasp=False)
        self.probe=dict(reference_parquet_sha256='reference',diagnostic_only=True,policy_inference=False,
            source_frame_playback=True,object_force_assistance_N=0.,object_torque_assistance_Nm=0.,
            grasps=[dict(passed=False)],all_grasps_passed=False)

    def test_authorized_pilot_can_learn_from_failed_grasp_without_becoming_complete(self):
        result=decide(self.cfg,self.policy,self.probe,'reference')
        self.assertTrue(result['training_admitted'])
        self.assertFalse(result['pretraining_grasp_passed'])
        self.assertTrue(result['post_training_grasp_acceptance_required'])
        self.assertTrue(result['nonpenetrating_reference_required'])
        self.assertFalse(result['strict_completion_granted'])

    def test_default_and_out_of_scope_and_nonexperimental_keep_original_gate(self):
        for cfg,policy in [(self.cfg,{}),(self.cfg,dict(self.policy,episodes=[1])),
                           (dict(self.cfg,experimental_training=False),self.policy),
                           (self.cfg,dict(self.policy,require_successful_pretraining_grasp=True))]:
            with self.subTest(cfg=cfg,policy=policy),self.assertRaises(AssertionError):
                decide(cfg,policy,self.probe,'reference')

    def test_stale_assisted_missing_or_wrong_timing_probe_is_still_rejected(self):
        for change in [dict(reference_parquet_sha256='old'),dict(object_force_assistance_N=1.),
                       dict(object_torque_assistance_Nm=1.),dict(source_frame_playback=False),dict(grasps=[])]:
            with self.subTest(change=change),self.assertRaises(AssertionError):
                decide(self.cfg,self.policy,dict(self.probe,**change),'reference')

    def test_original_mode_with_successful_grasp_still_works(self):
        result=decide(dict(self.cfg,experimental_training=False),{},dict(self.probe,all_grasps_passed=True),'reference')
        self.assertTrue(result['training_admitted'])


if __name__=='__main__':unittest.main()
