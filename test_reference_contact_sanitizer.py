import tempfile,unittest
from pathlib import Path
import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq
from fast_policy_execution import sanitize_reference_contacts
from improvement_common import digest


class ReferenceContactRepairTests(unittest.TestCase):
    def setup_reference(self,root):
        cfg=dict(training_reference='own',sequence='episode_000020')
        path=root/'human_motion_data/own/processed/sequence_id=episode_000020/robot_name=sharpa_wave/motion.parquet';path.parent.mkdir(parents=True)
        (root/'research').mkdir();(root/'flash_chord_coherent_actor_transfer').mkdir()
        row=dict(fps=20.,object_body_position=[[[.1,.2,.3]]])
        for side in ['left','right']:
            prefix=f'mano_{side}_object_contact_'
            row[prefix+'positions']=np.ones((1,5,3)).tolist();row[prefix+'part_ids']=[[1]*5]
            row[prefix+'normals']=[[[0.,0.,0.],[0.,0.,2.],[1.,0.,0.],[1.,0.,0.],[1.,0.,0.]]]
        pq.write_table(pa.Table.from_pylist([row]),path)
        return cfg,path,row

    def test_undefined_contacts_are_omitted_without_editing_poses_or_timing(self):
        with tempfile.TemporaryDirectory() as directory:
            root=Path(directory);cfg,path,before=self.setup_reference(root)
            self.assertTrue(sanitize_reference_contacts(root,cfg))
            after=pq.ParquetFile(path).read().to_pylist()[0]
            self.assertEqual(after['object_body_position'],before['object_body_position']);self.assertEqual(after['fps'],20.)
            for side in ['left','right']:
                prefix=f'mano_{side}_object_contact_'
                self.assertEqual(after[prefix+'part_ids'][0][0],0)
                self.assertEqual(after[prefix+'positions'][0][0],[0.,0.,0.])
                self.assertEqual(after[prefix+'normals'][0][1],[0.,0.,1.])
            self.assertFalse(sanitize_reference_contacts(root,cfg))

    def test_trained_references_cannot_be_rewritten(self):
        with tempfile.TemporaryDirectory() as directory:
            root=Path(directory);cfg,path,_=self.setup_reference(root);before=digest(path)
            (root/'flash_chord_coherent_actor_transfer/policy_32000.safetensors').write_bytes(b'trained')
            with self.assertRaisesRegex(AssertionError,'Never edit trained'):sanitize_reference_contacts(root,cfg)
            self.assertEqual(digest(path),before)
