import hashlib
import json
from pathlib import Path
import struct
import tempfile
import unittest
from actor_transfer_training import metadata,recipe


class ActorRecipeTests(unittest.TestCase):
    def test_recipe_preserves_the_seed_and_records_its_original_training(self):
        with tempfile.TemporaryDirectory() as directory:
            path=Path(directory)/'runs/episode_000038_newton_batch_20261015/flash_chord_coherent_actor_transfer/policy_90000384.safetensors'
            path.parent.mkdir(parents=True)
            header=json.dumps({'__metadata__':{'user.checkpoint_kind':'policy','user.algorithm':'flash_sac',
                                              'user.environment_steps':'90000384'}}).encode()
            path.write_bytes(struct.pack('<Q',len(header))+header)
            before=path.read_bytes();description=recipe(38,directory)
            self.assertEqual(description['source_checkpoint_sha256'],hashlib.sha256(before).hexdigest())
            self.assertEqual(description['source_environment_steps'],90000384)
            self.assertEqual(description['components'],['actor.params','actor.batch_stats'])
            self.assertTrue(description['fresh_critics'])
            self.assertTrue(description['training_counter_starts_at_zero'])
            self.assertEqual(path.read_bytes(),before)
            self.assertIsNone(recipe(25,directory))

    def test_invalid_checkpoint_header_is_rejected(self):
        with tempfile.TemporaryDirectory() as directory:
            path=Path(directory)/'bad';path.write_bytes(struct.pack('<Q',2**32))
            with self.assertRaisesRegex(ValueError,'header'):metadata(path)
