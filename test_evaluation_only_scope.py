import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import continuous_human_match_20261023 as controller
import public_validation_pipeline as public
import source_cad_queue as cad


class EvaluationOnlyScopeTests(unittest.TestCase):
    def test_existing_enabled_policy_still_requires_public_validation(self):
        self.assertTrue(controller.public_validation_required({'enabled': True}))
        for value in [None, 0, '', 'false']:
            self.assertTrue(controller.public_validation_required(
                {'enabled': True, 'public_validation_required': value}))

    def test_only_explicit_opt_out_skips_public(self):
        self.assertFalse(controller.public_validation_required(
            {'enabled': True, 'public_validation_required': False}))

    def test_evaluation_only_dispatches_existing_source_gate_without_public_worker(self):
        with tempfile.TemporaryDirectory() as directory:
            out = Path(directory)
            (out/'validated-pipeline-policy.json').write_text(json.dumps(
                dict(enabled=True, public_validation_required=False)))
            (out/'source-first-policy.json').write_text('{"enabled":true}')
            with patch.object(controller, 'OUT', out), \
                 patch.object(public, 'worker') as public_worker, \
                 patch.object(cad, 'run', return_value=False) as source_gate:
                controller.main()
                public_worker.assert_not_called()
                source_gate.assert_called_once_with(out, controller.FOCUS)


if __name__ == '__main__':
    unittest.main()
