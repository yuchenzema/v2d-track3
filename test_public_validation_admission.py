import unittest
import tempfile,json
from unittest.mock import patch
from pathlib import Path
from improvement_common import digest
from public_validation_pipeline import references_passed,masks_match_prompts


class PublicAdmissionTests(unittest.TestCase):
    def test_rejected_refinement_cannot_admit_a_weaker_retained_reference(self):
        rejected=dict(refined_metrics={'AUC':.99,'MP-SR':1.},best_reference_metrics={'AUC':.8,'MP-SR':0.},adopted_over_baseline=False)
        self.assertFalse(references_passed({'0':rejected},[0],.9))
        rejected['best_reference_metrics']={'AUC':.95,'MP-SR':1.}
        rejected['refined_metrics']={'AUC':.7,'MP-SR':0.}
        self.assertTrue(references_passed({'0':rejected},[0],.9))

    def test_queued_source_revision_blocks_admission_using_a_previous_source(self):
        result=self.good();result['source_revision']='_old'
        self.assertFalse(references_passed({'2':result},[2],.9,{'2':'_holes2'}))
        result['source_revision']='_holes2'
        self.assertTrue(references_passed({'2':result},[2],.9,{'2':'_holes2'}))

    def test_reviewed_masks_refresh_depth_with_serializable_command_and_bind_inputs(self):
        from public_validation_pipeline import reviewed_masks
        with tempfile.TemporaryDirectory() as directory:
            base=Path(directory);root=base/'episode';rec=root/'reconstruction';(rec/'masks/1').mkdir(parents=True)
            prior=base/'runs/episode_000005_newton_20261015/research';prior.mkdir(parents=True)
            (prior/'sam2-command.json').write_text('["docker","run"]')
            (root/'episode-config.json').write_text(json.dumps(dict(frames=1,objects=['white_pot'],camera_odometry_method='sift_pnp_metric',dataset_split='public')))
            prompts=rec/'sam2_prompts.json';prompts.write_text('{"prompts":[]}')
            (rec/'source-prompt-review.json').write_bytes(prompts.read_bytes())
            (rec/'masks/1/000000.png').write_bytes(b'mask')
            (rec/'masks/propagation-report.json').write_text(json.dumps(dict(prompts_sha256=digest(prompts))))
            alignment=rec/'stereo-depth-alignment.json';alignment.write_text('{"source_masks_prompt_sha256":"stale"}')
            def rerun(args,log):
                json.dumps(args) # batch_worker.run serializes command arguments before launching
                self.assertTrue(args[1].endswith('/scripts/dense_stereo_depth.py'))
                alignment.write_text('{"depth_recomputed":true}')
            with patch('public_validation_pipeline.BASE',base),patch('public_validation_pipeline.ROOT',root),patch('batch_worker.run',side_effect=rerun) as run,patch('batch_worker.docker') as docker,patch('source_mask_quality.inspect_and_repair'):
                reviewed_masks()
            run.assert_called_once();docker.assert_not_called()
            report=json.loads(alignment.read_text())
            self.assertTrue(report['depth_recomputed'])
            self.assertEqual(report['source_masks_prompt_sha256'],digest(prompts))

    def test_masks_from_previous_prompts_cannot_be_reused_after_hint_edit(self):
        with tempfile.TemporaryDirectory() as directory:
            root=Path(directory);rec=root/'reconstruction';(rec/'masks/1').mkdir(parents=True)
            (root/'episode-config.json').write_text(json.dumps(dict(frames=1,objects=['white_pot'])))
            prompt=rec/'sam2_prompts.json';prompt.write_text('{"prompts":[]}')
            (rec/'masks/1/000000.png').write_bytes(b'mask')
            (rec/'masks/propagation-report.json').write_text(json.dumps(dict(prompts_sha256=digest(prompt))))
            self.assertTrue(masks_match_prompts(root))
            prompt.write_text('{"prompts":[{"corrected_foreground_point":true}]}')
            self.assertFalse(masks_match_prompts(root))

    def test_measured_cap_mask_change_refreshes_depth_even_when_prompts_are_unchanged(self):
        from public_validation_pipeline import reviewed_masks
        with tempfile.TemporaryDirectory() as directory:
            base=Path(directory);root=base/'episode';rec=root/'reconstruction';(rec/'masks/1').mkdir(parents=True)
            (root/'research').mkdir();prior=base/'runs/episode_000005_newton_20261015/research';prior.mkdir(parents=True)
            (prior/'sam2-command.json').write_text('["docker","run"]')
            (root/'episode-config.json').write_text(json.dumps(dict(frames=1,objects=['white_pot'],camera_odometry_method='sift_pnp_metric')))
            prompts=rec/'sam2_prompts.json';prompts.write_text('{"prompts":[]}')
            mask=rec/'masks/1/000000.png';mask.write_bytes(b'new measured source cap foreground')
            (rec/'masks/propagation-report.json').write_text(json.dumps(dict(prompts_sha256=digest(prompts))))
            cap_masks={'masks/1/000000.png':digest(mask)}
            (root/'research/source-cap-mask-refinement.json').write_text(json.dumps(dict(masks_sha256=cap_masks)))
            alignment=rec/'stereo-depth-alignment.json'
            alignment.write_text(json.dumps(dict(source_masks_prompt_sha256=digest(prompts),source_cap_masks_sha256={'masks/1/000000.png':'old'})))
            def recompute(args,log):alignment.write_text('{"depth_recomputed":true}')
            with patch('public_validation_pipeline.BASE',base),patch('public_validation_pipeline.ROOT',root),patch('batch_worker.run',side_effect=recompute) as run,patch('source_mask_quality.inspect_and_repair'):
                reviewed_masks();reviewed_masks()
            run.assert_called_once()
            self.assertEqual(json.loads(alignment.read_text())['source_cap_masks_sha256'],cap_masks)

    def good(self):return dict(refined_metrics={'AUC':.95,'MP-SR':1.})
    def test_incomplete_public_roster_never_admits_evaluation(self):
        self.assertFalse(references_passed({'0':self.good()},[0,2],.9))
        self.assertFalse(references_passed({},[0,2],.9))
    def test_nonfinite_quality_is_not_a_pass(self):
        self.assertFalse(references_passed({'0':dict(refined_metrics={'AUC':float('nan'),'MP-SR':1.})},[0],.9))
    def test_complete_verified_reference_roster_meets_only_reference_gate(self):
        self.assertTrue(references_passed({'0':self.good(),'2':self.good()},[0,2],.9))
