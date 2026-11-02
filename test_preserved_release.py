import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch
import preserved_release as release


class PreservedReleaseTests(unittest.TestCase):
    def test_incomplete_source_follows_its_bound_original_baseline(self):
        with tempfile.TemporaryDirectory() as directory:
            base=Path(directory);source=base/'partial';source.mkdir();original=base/'original';(original/'flash_chord').mkdir(parents=True)
            actor=original/'flash_chord/policy_123.safetensors';actor.write_bytes(b'original actor')
            (source/'episode-config.json').write_text(json.dumps(dict(baseline_run=str(original))))
            with patch.object(release,'BATCH',base):checks=release.original_artifacts(source,25)
            self.assertIn(str(actor),checks);self.assertIn(str(source/'episode-config.json'),checks)
    def test_missing_legacy_manifest_freezes_actual_checkpoint_and_products(self):
        with tempfile.TemporaryDirectory() as directory:
            root=Path(directory)/'run';(root/'evaluation').mkdir(parents=True);(root/'flash_chord').mkdir()
            state=root/'evaluation/states.npz';state.write_bytes(b'original physics')
            actor=root/'flash_chord/policy_123.safetensors';actor.write_bytes(b'original policy')
            with patch.object(release,'BATCH',Path(directory)):
                watched=release.original_artifacts(root,25)
            self.assertIn(str(state),watched);self.assertIn(str(actor),watched)
            self.assertEqual(state.read_bytes(),b'original physics')

    def test_existing_manifest_hash_failure_is_not_silently_replaced(self):
        with tempfile.TemporaryDirectory() as directory:
            root=Path(directory);(root/'deliverables').mkdir();(root/'policy').write_bytes(b'changed')
            (root/'deliverables/manifest.json').write_text(json.dumps({'policy':dict(path='policy',sha256='0'*64)}))
            with self.assertRaisesRegex(RuntimeError,'changed'):release.original_artifacts(root,25)
