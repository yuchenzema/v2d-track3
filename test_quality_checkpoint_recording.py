import json,tempfile,unittest
from pathlib import Path
from unittest.mock import patch
import numpy as np
from fast_policy_execution import evaluate_checkpoint
from improvement_common import digest


class QualityCheckpointRecordingTests(unittest.TestCase):
    def test_replay_reuse_preserves_actual_counter_artifacts_and_receipt_identity(self):
        with tempfile.TemporaryDirectory() as directory:
            root=Path(directory);(root/'episode-config.json').write_text(json.dumps(dict(episode=1,frames=2,quality_training={})))
            # Enable quality archiving; an empty dict deliberately is not an enabled profile.
            cfg=json.loads((root/'episode-config.json').read_text());cfg['quality_training']={'world_count':128};(root/'episode-config.json').write_text(json.dumps(cfg))
            checkpoint=root/'policy_640000.safetensors';checkpoint.write_bytes(b'real-checkpoint-placeholder-for-test')
            trajectory=root/'actual.parquet';trajectory.write_bytes(b'actual-recording-placeholder-for-test')
            (root/'coherent-input-checksums.json').write_text('{"reference":"frozen"}')
            states=root/'states.npz'
            np.savez(states,policy_inference=True,checkpoint_sha256=digest(checkpoint),object_assistance=np.zeros(2),body_poses=np.zeros((2,1,7)))
            suite=dict(recorded_trajectory=str(trajectory),recorded_body_states=str(states),official_entrypoint_sha256='official',attempts=dict(recorded=dict(actual_training_steps=640000,metrics={'chord_sr':1.})))
            with patch('official_policy_suite.evaluate_candidate',return_value=suite),patch('official_policy_suite.verify_trajectory',return_value={'flash_chord.step_count':'2'}),patch('fast_policy_execution.utc',side_effect=['first','second','third']):
                a=evaluate_checkpoint(root,checkpoint,'native');b=evaluate_checkpoint(root,checkpoint,'native')
                self.assertEqual(a,b);self.assertEqual(b['actual_training_steps'],640000);self.assertEqual(b['updated_utc'],'first')
                self.assertFalse(b['strict_completion_granted']);self.assertEqual(b['states_sha256'],digest(states))
                np.savez(states,policy_inference=True,checkpoint_sha256=digest(checkpoint),object_assistance=np.ones(2),body_poses=np.zeros((2,1,7)))
                with self.assertRaises(AssertionError):evaluate_checkpoint(root,checkpoint,'native')
