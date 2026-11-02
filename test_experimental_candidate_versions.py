from pathlib import Path
import tempfile
import json
import unittest
from unittest.mock import patch
import evaluation_experiment_queue as queue


class CandidateVersionTests(unittest.TestCase):
    def test_exact_repaired_parent_uses_admitted_physical_derivative(self):
        with tempfile.TemporaryDirectory() as directory:
            out=Path(directory);candidate=out/'physical.json'
            candidate.write_text(json.dumps(dict(episode=38,trajectory_sha256='physical',
                source_candidate_parent_trajectory_sha256='original',
                reference_kind='source_revalidated_physical_candidate',physical_projection_applied=True)))
            (out/'experimental-source-candidates.json').write_text(json.dumps({'38':str(candidate)}))
            (out/'source-cad-status.json').write_text(json.dumps(dict(results={'38':dict(episode=38,trajectory_sha256='original')})))
            with patch.object(queue,'assess',return_value=dict(admitted=True,minimum_qualified_fraction=.9)):
                self.assertEqual([r[1]['trajectory_sha256'] for r in queue.candidates(out,[38],{})],['physical'])
                (out/'source-cad-status.json').write_text(json.dumps(dict(results={'38':dict(episode=38,trajectory_sha256='new')})))
                self.assertEqual({r[1]['trajectory_sha256'] for r in queue.candidates(out,[38],{})},{'physical','new'})

    def test_finished_source_cannot_resume_frozen_experiment(self):
        previous=dict(request='old.json',trajectory_sha256='same',exit_code=0)
        self.assertIsNone(queue.resumable_request(previous,previous,
            dict(status='pilot_finished',request='old.json'),38,'same'))
        self.assertIsNone(queue.resumable_request(previous,previous,
            dict(status='running',request='old.json'),38,'same'))

    def test_only_interrupted_current_job_with_same_inputs_resumes(self):
        with tempfile.TemporaryDirectory() as directory:
            request=Path(directory)/'new.json'
            request.write_text(json.dumps(dict(episode=38,source_version='same')))
            previous=dict(request='old.json',trajectory_sha256='same',exit_code=0)
            status=dict(status='running',request=str(request),execution_input_hashes={'physics':'v2'})
            with patch.object(queue,'execution_inputs',return_value={'physics':'v2'}):
                self.assertEqual(queue.resumable_request(previous,previous,status,38,'same'),request)
            with patch.object(queue,'execution_inputs',return_value={'physics':'v3'}):
                self.assertIsNone(queue.resumable_request(previous,previous,status,38,'same'))

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
