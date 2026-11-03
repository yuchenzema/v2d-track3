import json,tempfile,unittest
from pathlib import Path
from unittest.mock import patch
import numpy as np
from safetensors.numpy import save_file
from improvement_common import save,digest
from own_submission_queue import eligible_fast


class ArchivedWholeCheckpointTests(unittest.TestCase):
    def test_archived_policy_is_bound_to_its_real_counter_and_full_recording(self):
        with tempfile.TemporaryDirectory() as directory:
            root=Path(directory);folder=root/'flash_chord_coherent_actor_transfer';folder.mkdir()
            checkpoint=folder/'policy_640000.safetensors';save_file({'actor':np.ones(2,dtype=np.float32)},str(checkpoint),metadata={'user.environment_steps':'640000'})
            sha=digest(checkpoint);trajectory=root/'actual.parquet';trajectory.write_bytes(b'official-recording')
            states=root/'states.npz';poses=np.zeros((2,1,7));poses[:,:,3]=1.
            np.savez(states,policy_inference=True,checkpoint_sha256=sha,object_assistance=np.zeros(2),body_poses=poses,reference_frames=np.arange(2))
            (root/'reference').write_bytes(b'frozen');save(root/'coherent-input-checksums.json',{'reference':digest(root/'reference')})
            save(root/f'evaluation/official/{sha[:16]}/suite.json',dict(status='official_evaluation_complete',checkpoint_sha256=sha,official_entrypoint_sha256='official',recorded_trajectory=str(trajectory),recorded_body_states=str(states),attempts={'recorded':{'actual_training_steps':640000}}))
            result=dict(status='ready_for_submission',public_object_scans_used=False,checkpoint=str(checkpoint),checkpoint_sha256=sha,actual_training_steps=640000,
                trajectory_sha256=digest(trajectory),states_sha256=digest(states),frozen_input_manifest_sha256=digest(root/'coherent-input-checksums.json'))
            archive=root/'research/quality-trained-candidates/policy.json';save(archive,result);cfg=dict(episode=1,frames=2,objects=['object'])
            with patch('official_policy_suite.verify_trajectory',return_value={'flash_chord.step_count':'2'}):
                item=eligible_fast(root,cfg,archive);self.assertEqual(item['checkpoint'],str(checkpoint));self.assertEqual(item['result_override'],str(archive))
                result['actual_training_steps']=1280000;save(archive,result)
                with self.assertRaisesRegex(ValueError,'filename'):eligible_fast(root,cfg,archive)
