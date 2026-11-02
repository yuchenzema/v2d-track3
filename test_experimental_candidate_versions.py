from pathlib import Path
import tempfile
import json
import unittest
from unittest.mock import patch
import evaluation_experiment_queue as queue


class CandidateVersionTests(unittest.TestCase):
    def test_registered_old_source_cannot_hide_a_new_admitted_reconstruction(self):
        with tempfile.TemporaryDirectory() as d:
            out=Path(d);old=out/'old.json'
            old.write_text(json.dumps(dict(trajectory_sha256='old',episode=38)))
            (out/'experimental-source-candidates.json').write_text(json.dumps({'38':str(old)}))
            (out/'source-cad-status.json').write_text(json.dumps(dict(results={'38':dict(trajectory_sha256='new',episode=38)})))
            (out/'experimental-training-status.json').write_text(json.dumps(dict(attempts={'38':dict(trajectory_sha256='old',execution_input_hashes={'experiment':'same','evaluation_experiment_queue.py':'old scheduler'})})))
            with patch.object(queue,'assess',return_value=dict(admitted=True,minimum_qualified_fraction=.9)),\
                 patch.object(queue,'execution_inputs',return_value={'experiment':'same','evaluation_experiment_queue.py':'new scheduler'}):
                self.assertEqual([r[1]['trajectory_sha256'] for r in queue.candidates(out,[38],{})],['new'])
            state=json.loads((out/'experimental-training-status.json').read_text())
            state['attempts']['38']['source_attempts']={'new':dict(execution_input_hashes={'experiment':'same'})}
            (out/'experimental-training-status.json').write_text(json.dumps(state))
            with patch.object(queue,'assess',return_value=dict(admitted=True,minimum_qualified_fraction=.9)),\
                 patch.object(queue,'execution_inputs',return_value={'experiment':'same'}):
                self.assertEqual(queue.candidates(out,[38],{}),[])
