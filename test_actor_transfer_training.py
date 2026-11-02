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

    def test_a_static_or_unbound_candidate_cannot_replace_the_seed(self):
        with tempfile.TemporaryDirectory() as directory:
            root=Path(directory);old=root/'runs/episode_000038_newton_batch_20261015/flash_chord_coherent_actor_transfer/policy_90000384.safetensors'
            old.parent.mkdir(parents=True)
            def checkpoint(path,steps):
                header=json.dumps({'__metadata__':{'user.checkpoint_kind':'policy','user.algorithm':'flash_sac','user.environment_steps':str(steps)}}).encode()
                path.write_bytes(struct.pack('<Q',len(header))+header)
            checkpoint(old,90000384)
            current=root/'runs/episode_000038_newton_human_match_20261023_g007';(current/'research').mkdir(parents=True);(current/'evaluation').mkdir()
            candidate=current/'actor.safetensors';checkpoint(candidate,1280000);sha=hashlib.sha256(candidate.read_bytes()).hexdigest()
            metrics=dict(checkpoint='/episode/actor.safetensors',checkpoint_sha256=sha,metrics=dict(object_position_error_m=.04))
            grasp=dict(policy_inference=True,source_frames_checked=263,checkpoint_sha256=sha,
                       grasps=[dict(expected_motion_m=.24,actual_motion_m=.006)])
            strict=dict(all_frames_checked=True,source_frames=263,checkpoint_sha256=sha,post_training_grasp_acceptance=grasp)
            (current/'evaluation/final-metrics.json').write_text(json.dumps(metrics));(current/'research/strict-human-match.json').write_text(json.dumps(strict))
            self.assertEqual(recipe(38,root)['source_environment_steps'],90000384)
            grasp['grasps'][0]['actual_motion_m']=.23
            (current/'research/strict-human-match.json').write_text(json.dumps(strict))
            chosen=recipe(38,root);self.assertEqual(chosen['source_environment_steps'],1280000)
            self.assertFalse(chosen['source_episode_strictly_passed'])
            strict['checkpoint_sha256']='mismatched'
            (current/'research/strict-human-match.json').write_text(json.dumps(strict))
            self.assertEqual(recipe(38,root)['source_environment_steps'],90000384)
