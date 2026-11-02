import json
from pathlib import Path
import tempfile
import unittest
from improvement_common import digest
from training_continuation import extend


class OwnTrainingContinuationTests(unittest.TestCase):
    def fixture(self,base):
        folder=base/'flash_chord_coherent_actor_transfer';folder.mkdir()
        (folder/'policy_100.safetensors').write_bytes(b'own actor')
        (folder/'state_100.safetensors').write_bytes(b'own optimizer state')
        (folder/'TRAINING_COMPLETED').touch();(base/'reference').write_bytes(b'frozen own reference')
        (base/'coherent-input-checksums.json').write_text(json.dumps({'reference':digest(base/'reference')}))
        (base/'episode-config.json').write_text(json.dumps(dict(experimental_training=True,own_geometry_required=True,
            own_training_continuation_allowed=True,own_training_maximum_steps=180,
            validated_video_pipeline=True,training_curriculum='fresh_rebuild',training_budget=100)))
        return folder

    def test_own_continuation_restores_learner_and_respects_authorized_cap(self):
        with tempfile.TemporaryDirectory() as directory:
            base=Path(directory);folder=self.fixture(base)
            self.assertEqual(extend(base,increment=100),180)
            record=json.loads((base/'research/training-continuation-001.json').read_text())
            self.assertEqual(record['resume_environment_steps'],100)
            self.assertEqual(record['additional_environment_steps_requested'],80)
            self.assertTrue(record['optimizer_restore_required'])
            self.assertFalse((folder/'TRAINING_COMPLETED').exists())

    def test_reference_change_blocks_continuation_before_marker_is_removed(self):
        with tempfile.TemporaryDirectory() as directory:
            base=Path(directory);folder=self.fixture(base);(base/'reference').write_bytes(b'changed')
            with self.assertRaisesRegex(AssertionError,'Frozen reference changed'):extend(base,increment=100)
            self.assertTrue((folder/'TRAINING_COMPLETED').exists())
