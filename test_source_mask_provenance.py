import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

from improvement_common import digest
from source_cad_alignment import verify_current_source_masks


class SourceMaskProvenanceTests(unittest.TestCase):
    def inputs(self,root):
        rec=root/'reconstruction';(rec/'masks').mkdir(parents=True)
        prompts=rec/'sam2_prompts.json';prompts.write_text('{"prompts":[]}')
        review=rec/'source-prompt-review.json';review.write_bytes(prompts.read_bytes())
        (rec/'masks/propagation-report.json').write_text(json.dumps(dict(prompts_sha256=digest(prompts))))
        return rec,prompts,review

    def test_new_review_cannot_be_silently_skipped_by_old_completion_markers(self):
        with tempfile.TemporaryDirectory() as directory:
            root=Path(directory);rec,_,review=self.inputs(root)
            (rec/'masks-reviewed.json').write_text('{"completed":true}')
            (rec/'scene_motion_physical.npz').write_bytes(b'old object poses must not admit CAD')
            review.write_text('{"prompts":[{"frame_index":274}]}')
            with self.assertRaisesRegex(RuntimeError,'fresh frontend revision'):
                verify_current_source_masks(root,dict(validated_video_pipeline=True))

    def test_updated_prompt_without_new_propagation_is_rejected(self):
        with tempfile.TemporaryDirectory() as directory:
            root=Path(directory);_,prompts,review=self.inputs(root)
            prompts.write_text('{"prompts":[{"frame_index":274}]}');review.write_bytes(prompts.read_bytes())
            with self.assertRaisesRegex(RuntimeError,'after propagation'):
                verify_current_source_masks(root,dict(validated_video_pipeline=True))

    def test_only_descriptive_provenance_edits_do_not_invalidate_identical_conditioning(self):
        with tempfile.TemporaryDirectory() as directory:
            root=Path(directory);_,prompts,review=self.inputs(root)
            review.write_text('{"prompts":[],"provenance":"Original source review"}')
            result=verify_current_source_masks(root,dict(validated_video_pipeline=True))
            self.assertEqual(result['source_prompts_sha256'],digest(prompts))
            self.assertEqual(result['source_review_sha256'],digest(review))

    def test_consistent_source_binding_is_required_and_recorded(self):
        with tempfile.TemporaryDirectory() as directory:
            root=Path(directory);rec,prompts,_=self.inputs(root)
            result=verify_current_source_masks(root,dict(validated_video_pipeline=True))
            self.assertEqual(result['source_review_sha256'],digest(prompts))
            self.assertEqual(result['propagation_prompts_sha256'],digest(prompts))
            (rec/'masks/propagation-report.json').unlink()
            with self.assertRaisesRegex(RuntimeError,'provenance'):
                verify_current_source_masks(root,dict(validated_video_pipeline=True))

    def test_an_ignored_instance_ownership_request_cannot_admit_legacy_binary_masks(self):
        with tempfile.TemporaryDirectory() as directory:
            root=Path(directory);rec,prompts,review=self.inputs(root)
            prompts.write_text('{"prompts":[],"exclusive_visible_instances":true}');review.write_bytes(prompts.read_bytes())
            path=rec/'masks/propagation-report.json';report=dict(prompts_sha256=digest(prompts));path.write_text(json.dumps(report))
            with self.assertRaisesRegex(RuntimeError,'ownership metadata'):
                verify_current_source_masks(root,dict(validated_video_pipeline=True))
            report.update(visible_instance_ownership='highest measured SAM logit at overlapping foreground pixels',
                          overlapping_foreground_pixels_after_ownership=0,ownership_adds_no_source_foreground=True)
            path.write_text(json.dumps(report))
            self.assertEqual(verify_current_source_masks(root,dict(validated_video_pipeline=True))['source_review_sha256'],digest(prompts))

    def test_cross_instance_conditioning_cannot_be_silently_ignored(self):
        with tempfile.TemporaryDirectory() as directory:
            root=Path(directory);rec,prompts,review=self.inputs(root)
            prompts.write_text('{"prompts":[],"cross_instance_negatives":true}');review.write_bytes(prompts.read_bytes())
            path=rec/'masks/propagation-report.json';report=dict(prompts_sha256=digest(prompts));path.write_text(json.dumps(report))
            with self.assertRaisesRegex(RuntimeError,'cross-instance conditioning'):
                verify_current_source_masks(root,dict(validated_video_pipeline=True))
            report['cross_instance_negative_conditioning_requested']=True;path.write_text(json.dumps(report))
            self.assertEqual(verify_current_source_masks(root,dict(validated_video_pipeline=True))['source_prompts_sha256'],digest(prompts))

    def test_cap_masks_with_stale_metric_depth_cannot_admit_derived_object_poses(self):
        with tempfile.TemporaryDirectory() as directory:
            root=Path(directory);rec,_,_=self.inputs(root);(root/'research').mkdir()
            mask_hashes={'masks/1/000000.png':'current observed cap mask'}
            report=root/'research/source-cap-mask-refinement.json';report.write_text(json.dumps(dict(masks_sha256=mask_hashes)))
            with patch('source_cap_masks.verify',return_value=dict(path=str(report),sha256=digest(report))):
                with self.assertRaisesRegex(RuntimeError,'after metric depth'):
                    verify_current_source_masks(root,dict(validated_video_pipeline=True))
                (rec/'stereo-depth-alignment.json').write_text(json.dumps(dict(source_cap_masks_sha256=mask_hashes)))
                self.assertIn('source_cap_refinement',verify_current_source_masks(root,dict(validated_video_pipeline=True)))
