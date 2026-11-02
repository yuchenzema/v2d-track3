import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

from improvement_common import digest
from public_validation_pipeline import cad_repair_input_changes


class PublicCadRetryInputTests(unittest.TestCase):
    def fixture(self,base):
        out=base/'queue';out.mkdir()
        source=base/'source_marker_pose_proposals.py';source.write_text('repaired algorithm')
        research=base/'latest/research';research.mkdir(parents=True)
        report=research/'cad-alignment-report.json'
        report.write_text(json.dumps(dict(algorithm_files_sha256={source.name:digest(source)})))
        return out,source,research,dict(repair_round=3,refined_run=str(research.parent))

    def test_unchanged_exhausted_attempt_is_not_repeated(self):
        with tempfile.TemporaryDirectory() as directory:
            base=Path(directory);out,_,_,previous=self.fixture(base)
            with patch('public_validation_pipeline.BASE',base),patch('public_validation_pipeline.OUT',out):
                self.assertEqual(cad_repair_input_changes(previous,23),[])

    def test_concrete_algorithm_repair_reopens_the_exhausted_episode(self):
        with tempfile.TemporaryDirectory() as directory:
            base=Path(directory);out,source,_,previous=self.fixture(base)
            source.write_text('new source-verified correction')
            with patch('public_validation_pipeline.BASE',base),patch('public_validation_pipeline.OUT',out):
                self.assertEqual(cad_repair_input_changes(previous,23),[source.name])

    def test_newest_rejected_attempt_prevents_repeated_retry_of_same_repair(self):
        with tempfile.TemporaryDirectory() as directory:
            base=Path(directory);out,source,_,latest=self.fixture(base)
            old=base/'retained/research';old.mkdir(parents=True)
            (old/'cad-alignment-report.json').write_text(json.dumps(dict(algorithm_files_sha256={source.name:'older hash'})))
            previous=dict(repair_round=3,refined_run=str(old.parent),rejected_refinements=[latest])
            with patch('public_validation_pipeline.BASE',base),patch('public_validation_pipeline.OUT',out):
                self.assertEqual(cad_repair_input_changes(previous,23),[])

    def test_source_review_repair_is_detected_then_consumed_by_native_binding(self):
        with tempfile.TemporaryDirectory() as directory:
            base=Path(directory);out,_,research,previous=self.fixture(base)
            (out/'source-marker-overrides').mkdir()
            review=out/'source-marker-overrides/public_000023.json';review.write_text('{"new_source_frame":155}')
            with patch('public_validation_pipeline.BASE',base),patch('public_validation_pipeline.OUT',out):
                self.assertEqual(cad_repair_input_changes(previous,23),['source-marker-review'])
                (research/'source-marker-review.json').write_text(json.dumps(dict(binding=dict(sha256=digest(review)))))
                self.assertEqual(cad_repair_input_changes(previous,23),[])

    def test_missing_provenance_does_not_invent_an_algorithm_change(self):
        with tempfile.TemporaryDirectory() as directory:
            base=Path(directory);out,_,research,previous=self.fixture(base)
            (research/'cad-alignment-report.json').unlink()
            with patch('public_validation_pipeline.BASE',base),patch('public_validation_pipeline.OUT',out):
                self.assertEqual(cad_repair_input_changes(previous,23),[])
