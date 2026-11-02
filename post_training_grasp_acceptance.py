"""Score grasp success from the actual unmodified official policy recording."""
import json
from pathlib import Path
import numpy as np
import pyarrow.parquet as pq
from improvement_common import digest,save,utc


def score(reference,achieved,grasps):
    assert reference.shape==achieved.shape and np.isfinite(achieved).all(),'Incomplete/nonfinite policy trajectory'
    rows=[]
    for index,g in enumerate(grasps):
        start,end,body=g['start'],g['end'],g['object']
        assert 0<=start<=end<len(reference)
        expected=reference[start:end+1,body];actual=achieved[start:end+1,body]
        error=float(np.max(np.linalg.norm(actual-expected,axis=1)))
        expected_motion=float(np.max(np.linalg.norm(expected-expected[0],axis=1)))
        actual_motion=float(np.max(np.linalg.norm(actual-actual[0],axis=1)))
        passed=error<=.02 and (expected_motion<.015 or actual_motion>=.75*expected_motion)
        rows.append(dict(grasp=index,object_index=body,source_start=start,source_end=end,
            maximum_object_position_error_m=error,expected_motion_m=expected_motion,
            actual_motion_m=actual_motion,passed=bool(passed)))
    return dict(grasps=rows,all_grasps_passed=bool(rows) and all(r['passed'] for r in rows))


def evaluate(root,suite):
    root=Path(root);cfg=json.loads((root/'episode-config.json').read_text())
    path=Path(suite['recorded_trajectory']);table=pq.read_table(path)
    meta=table.schema.metadata or {}
    assert meta.get(b'flash_chord.checkpoint_sha256')==suite['checkpoint_sha256'].encode(),'Wrong policy checkpoint'
    assert meta.get(b'flash_chord.eval_script_sha256')==suite['official_entrypoint_sha256'].encode(),'Wrong official evaluator'
    assert meta.get(b'flash_chord.world_count')==b'1' and meta.get(b'flash_chord.start_frame')==b'0'
    assert table['environment_valid'].to_numpy().all(),'Failed official policy world'
    reference_path=root/f"human_motion_data/{cfg['training_reference']}/processed/sequence_id={cfg['sequence']}/robot_name=sharpa_wave/motion.parquet"
    reference=np.asarray(pq.ParquetFile(reference_path).read(columns=['object_body_position']).to_pylist()[0]['object_body_position'])
    assert len(reference)==cfg['frames']
    names=np.asarray(table['object_body_name'].to_pylist());frames=table['reference_frame'].to_numpy()
    values=np.column_stack([table['achieved_position_'+axis].to_numpy() for axis in 'xyz'])
    achieved=np.empty_like(reference)
    for body,name in enumerate(cfg['objects']):
        ids=np.flatnonzero((names==name)|(names==f'track3_{name}/object'))
        assert len(ids)==len(reference) and np.array_equal(np.sort(frames[ids]),np.arange(len(reference))),'Missing/duplicate source frames'
        achieved[:,body]=values[ids[np.argsort(frames[ids])]]
    report=dict(score(reference,achieved,cfg['grasps']),updated_utc=utc(),policy_inference=True,
        checkpoint_sha256=suite['checkpoint_sha256'],official_trajectory_sha256=digest(path),
        reference_parquet_sha256=digest(reference_path),source_frames_checked=len(reference),
        diagnostic_only=False,strict_completion_granted=False,
        note='Object grasp motion checked on actual policy output; final collision/source/manual acceptance remains mandatory.')
    save(path.parent.parent/'grasp-acceptance.json',report)
    return report
