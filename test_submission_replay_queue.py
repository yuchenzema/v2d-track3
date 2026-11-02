import json
from pathlib import Path
import tempfile
import unittest

import pyarrow as pa
import pyarrow.parquet as pq
import submission_replay_queue as replay


class SubmissionReplayTests(unittest.TestCase):
    def test_blocked_request_requires_changed_inputs_before_retry(self):
        with tempfile.TemporaryDirectory() as d:
            out=Path(d);request=out/'submission-replay-request.json'
            request.write_text('{"enabled":true}')
            self.assertTrue(replay.pending(out))
            (out/'submission-replay-status.json').write_text(json.dumps(dict(status='blocked',request_sha256=replay.digest(request))))
            self.assertFalse(replay.pending(out))
            request.write_text('{"enabled":true,"repair_revision":2}')
            self.assertTrue(replay.pending(out))

    def test_reference_and_checkpoint_mounts_are_readonly_and_offline(self):
        command=replay.docker_command('/source','/policy','/kit','/runtime','/out',25)[-1]
        self.assertIn('/source:/episode:ro',command)
        self.assertIn('/policy:/checkpoint/policy.safetensors:ro',command)
        self.assertIn('--network none',command)
        self.assertIn('evaluation.world_count=1',command)
        self.assertNotIn('train_coherent_actor_transfer',command)
        self.assertNotIn('evaluation.motion_end_frame',command)

    def test_nested_recording_uses_actual_episode_root(self):
        with tempfile.TemporaryDirectory() as d:
            root=Path(d);(root/'episode-config.json').write_text('{}')
            self.assertEqual(replay.episode_root(root/'research/release/states.npz'),root)

    def trajectory(self,path,valid=True):
        values={f'achieved_{n}':[v] for n,v in zip(['position_x','position_y','position_z','quaternion_x','quaternion_y','quaternion_z','quaternion_w'],[0,0,0,0,0,0,1])}
        values['environment_valid']=[valid]
        meta={('flash_chord.'+k).encode():v.encode() for k,v in dict(episode_index='25',checkpoint_sha256='ck',eval_script_sha256='script',eval_code_sha256='code',world_count='1',start_frame='0',quaternion_order='xyzw').items()}
        pq.write_table(pa.table(values).replace_schema_metadata(meta),path)

    def test_provenance_and_invalid_environment_are_gates(self):
        with tempfile.TemporaryDirectory() as d:
            path=Path(d)/'rollout.parquet';self.trajectory(path)
            self.assertEqual(replay.verify_output(path,25,'ck','script','code')['rows'],1)
            with self.assertRaisesRegex(AssertionError,'provenance'):
                replay.verify_output(path,25,'different-policy','script','code')
            self.trajectory(path,valid=False)
            with self.assertRaisesRegex(AssertionError,'Non-finite'):
                replay.verify_output(path,25,'ck','script','code')
