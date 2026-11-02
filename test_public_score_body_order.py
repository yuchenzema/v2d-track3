"""Public metadata and per-frame object lists need not have identical order."""
import json
from pathlib import Path
import tempfile
import unittest
import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq
from scipy.spatial.transform import Rotation
from t3_local_score import PUBLIC,compute,load_public,prediction_body_order


class PublicBodyOrderTests(unittest.TestCase):
    def test_public18_metadata_order_is_reconciled_before_official_metrics(self):
        names,truth,valid,_=load_public(18)
        metadata=json.loads(pq.read_schema(PUBLIC/'data/chunk-000/episode_000018.parquet').metadata[b'objects'])
        self.assertNotEqual(names,metadata)
        permutation=[names.index(name) for name in metadata]
        matrices=np.broadcast_to(np.eye(4),(len(truth),len(names),4,4)).copy()
        matrices[:,:,:3,3]=truth[:,:,:3]
        matrices[:,:,:3,:3]=Rotation.from_quat(truth[:,:,3:].reshape(-1,4)).as_matrix().reshape(len(truth),len(names),3,3)
        with tempfile.TemporaryDirectory() as directory:
            root=Path(directory);(root/'reconstruction').mkdir()
            (root/'episode-config.json').write_text(json.dumps(dict(objects=metadata)))
            path=root/'reconstruction/control.npz';np.savez(path,object_to_world=matrices[:,permutation])
            result=compute(18,path,sample_count=100,include_mesh=False)
        self.assertEqual(result['prediction_object_order'],metadata)
        self.assertEqual(result['scoring_object_order'],names)
        self.assertAlmostEqual(result['metrics']['AUC'],1.,places=5)
        self.assertLess(result['metrics']['MPPE_cm'],1e-4)

    def test_named_arrays_and_unknown_or_duplicate_rosters(self):
        with tempfile.TemporaryDirectory() as directory:
            path=Path(directory)/'prediction.npz'
            order,names,source=prediction_body_order(path,dict(object_names=np.array(['pan','brush'])),['brush','pan'])
            self.assertEqual(order,[1,0])
            for names in [['pan','pan'],['pan','cup']]:
                with self.assertRaisesRegex(ValueError,'roster differs'):
                    prediction_body_order(path,dict(object_names=np.array(names)),['brush','pan'])

    def test_unlabelled_arrays_are_rejected(self):
        with tempfile.TemporaryDirectory() as directory:
            with self.assertRaisesRegex(ValueError,'object names missing'):
                prediction_body_order(Path(directory)/'prediction.npz',{},['brush','pan'])

    def test_policy_records_use_names_and_reject_changing_body_identity(self):
        names,truth,valid,_=load_public(18);rows=[]
        fields=['position_x','position_y','position_z','quaternion_x','quaternion_y','quaternion_z','quaternion_w']
        for frame in range(len(truth)):
            for recorded_body,name in enumerate(reversed(names)):
                row=dict(reference_frame=frame,object_body_id=recorded_body,
                         object_body_name='track3_'+name+'/body',environment_valid=True)
                row.update({'achieved_'+field:float(value) for field,value in zip(fields,truth[frame,names.index(name)])})
                rows.append(row)
        metadata={b'flash_chord.world_count':b'1',b'flash_chord.start_frame':b'0',
                  b'flash_chord.eval_script_sha256':b'unit-test-control',b'flash_chord.checkpoint_sha256':b'unit-test-control'}
        with tempfile.TemporaryDirectory() as directory:
            path=Path(directory)/'control.parquet'
            pq.write_table(pa.Table.from_pylist(rows).replace_schema_metadata(metadata),path)
            result=compute(18,path,sample_count=100,include_mesh=False)
            self.assertAlmostEqual(result['metrics']['AUC'],1.,places=5)
            self.assertEqual(result['prediction_object_order'],list(reversed(names)))
            rows[2]['object_body_name']=rows[1]['object_body_name']
            pq.write_table(pa.Table.from_pylist(rows).replace_schema_metadata(metadata),path)
            with self.assertRaisesRegex(AssertionError,'changes name'):
                compute(18,path,sample_count=100,include_mesh=False)


if __name__=='__main__':unittest.main()
