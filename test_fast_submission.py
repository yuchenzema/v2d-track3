import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch
import numpy as np
from improvement_common import digest
import own_submission_queue as queue


class FastSubmissionTests(unittest.TestCase):
    def fixture(self,root,assistance=0.,frames=None):
        (root/'research').mkdir();(root/'flash_chord_coherent_actor_transfer').mkdir()
        cfg=dict(episode=1,frames=2,objects=['object'],own_geometry_required=True,fast_submission=True,training_initial_counter=0)
        (root/'episode-config.json').write_text(json.dumps(cfg))
        (root/'flash_chord_coherent_actor_transfer/TRAINING_COMPLETED').touch()
        actor=root/'policy.safetensors';actor.write_bytes(b'actual checkpoint artifact');sha=digest(actor)
        folder=root/f'evaluation/official/{sha[:16]}';folder.mkdir(parents=True)
        trajectory=folder/'episode_000001.parquet';trajectory.write_bytes(b'official parquet artifact')
        states=folder/'states.npz';np.savez(states,body_poses=np.zeros((2,1,7)),object_assistance=[assistance,assistance],
            policy_inference=True,checkpoint_sha256=sha,reference_frames=np.arange(2) if frames is None else frames)
        source=root/'own-reference';source.write_bytes(b'frozen reference')
        frozen=root/'coherent-input-checksums.json';frozen.write_text(json.dumps({'own-reference':digest(source)}))
        (folder/'suite.json').write_text(json.dumps(dict(status='official_evaluation_complete',checkpoint_sha256=sha,
            recorded_trajectory=str(trajectory),recorded_body_states=str(states),official_entrypoint_sha256='official',
            attempts=dict(recorded=dict(actual_training_steps=640000)))))
        (root/'research/fast-submission-result.json').write_text(json.dumps(dict(status='ready_for_submission',
            checkpoint=str(actor),checkpoint_sha256=sha,trajectory_sha256=digest(trajectory),states_sha256=digest(states),
            public_object_scans_used=False,frozen_input_manifest_sha256=digest(frozen))))
        (root/'research/strict-human-match.json').write_text(json.dumps(dict(passes_strict_local_checks=False,
            failed_frame_count=2,contact_validation_failed=True)))
        return cfg

    def evaluate(self,root):
        with patch('own_geometry.validate'),patch('fast_submission_policy.enabled',return_value=True),\
             patch('official_policy_suite.verify_trajectory',return_value={'flash_chord.step_count':'2'}):
            return queue.eligible(root)

    def test_failed_motion_and_contact_quality_do_not_block_valid_complete_first_recording(self):
        with tempfile.TemporaryDirectory() as directory:
            root=Path(directory);self.fixture(root);before=(root/'research/strict-human-match.json').read_bytes()
            result=self.evaluate(root)
            self.assertFalse(result['quality_acceptance_required']);self.assertFalse(result['strict_completion_granted'])
            self.assertEqual((root/'research/strict-human-match.json').read_bytes(),before)

    def test_object_assistance_is_rejected_even_in_fast_mode(self):
        with tempfile.TemporaryDirectory() as directory:
            root=Path(directory);self.fixture(root,assistance=1.)
            with self.assertRaisesRegex(ValueError,'force assistance'):self.evaluate(root)

    def test_scrambled_source_clock_is_rejected_even_in_fast_mode(self):
        with tempfile.TemporaryDirectory() as directory:
            root=Path(directory);self.fixture(root,frames=np.array([1,0]))
            with self.assertRaisesRegex(ValueError,'frame order'):self.evaluate(root)

    def test_changed_reference_is_rejected_even_in_fast_mode(self):
        with tempfile.TemporaryDirectory() as directory:
            root=Path(directory);self.fixture(root);(root/'own-reference').write_bytes(b'edited')
            with self.assertRaisesRegex(ValueError,'Frozen'):self.evaluate(root)

    def test_physics_setup_can_change_only_before_any_learning(self):
        from fast_policy_execution import reset_untrained_setup
        for trained in [False,True]:
            with self.subTest(trained=trained),tempfile.TemporaryDirectory() as directory:
                root=Path(directory);self.fixture(root)
                (root/'flash_chord_coherent_actor_transfer/TRAINING_COMPLETED').unlink()
                (root/'scripts').mkdir();(root/'scripts/train_coherent_actor_transfer.py').write_text('old physics setup')
                (root/'coherent-training-config.json').write_text(json.dumps(dict(scene=dict(decompose_objects=True))))
                if trained:(root/'flash_chord_coherent_actor_transfer/metrics.jsonl').write_text(json.dumps(dict(environment_steps=512)))
                frozen=root/'coherent-input-checksums.json';before=frozen.read_bytes()
                self.assertEqual(reset_untrained_setup(root),not trained)
                if trained:self.assertEqual(frozen.read_bytes(),before)
                else:
                    self.assertFalse(frozen.exists())
                    self.assertEqual(next((root/'research/pretraining-setup-history').glob('*/coherent-input-checksums.json')).read_bytes(),before)

    def test_saved_policy_also_prevents_physics_setup_changes(self):
        from fast_policy_execution import reset_untrained_setup
        with tempfile.TemporaryDirectory() as directory:
            root=Path(directory);self.fixture(root)
            (root/'flash_chord_coherent_actor_transfer/TRAINING_COMPLETED').unlink()
            (root/'coherent-training-config.json').write_text(json.dumps(dict(scene=dict(decompose_objects=True))))
            (root/'flash_chord_coherent_actor_transfer/policy_0.safetensors').write_bytes(b'own policy')
            self.assertFalse(reset_untrained_setup(root))

    def test_physics_color_omission_preserves_geometry_and_submission_mesh(self):
        import trimesh
        from fast_policy_execution import prepare_physics_exports
        with tempfile.TemporaryDirectory() as directory:
            root=Path(directory);assets=root/'reconstruction/mesh_priors';assets.mkdir(parents=True);(root/'research').mkdir()
            mesh=trimesh.creation.icosphere(subdivisions=1)
            mesh.visual.vertex_colors=np.random.default_rng(1).integers(0,255,(len(mesh.vertices),4),dtype=np.uint8)
            for extension in ['ply','obj','glb']:mesh.export(assets/f'object.{extension}')
            hashes={extension:digest(assets/f'object.{extension}') for extension in ['ply','obj','glb']}
            path=root/'research/own-geometry-provenance.json';path.write_text(json.dumps(dict(objects=dict(object=dict(mesh_sha256=hashes)))))
            source_sha=digest(path);prepare_physics_exports(root,source_sha)
            proof=json.loads(path.read_text());self.assertFalse(proof['physics_export_changes_geometry'])
            self.assertEqual(digest(assets/'object.glb'),hashes['glb']);self.assertEqual(digest(assets/'object.ply'),hashes['ply'])
            vertices=[line for line in (assets/'object.obj').read_text().splitlines() if line.startswith('v ')]
            self.assertTrue(vertices);self.assertTrue(all(len(line.split())==4 for line in vertices))
