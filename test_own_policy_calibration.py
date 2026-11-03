import json,tempfile,unittest
from pathlib import Path
from unittest.mock import patch
import numpy as np
from safetensors import safe_open
from safetensors.numpy import save_file,load_file
from own_policy_calibration import derive,weights_digest


class OwnPolicyCalibrationTests(unittest.TestCase):
    def test_declared_gain_variant_preserves_weights_steps_and_original_limits(self):
        with tempfile.TemporaryDirectory() as directory:
            root=Path(directory);parent=root/'policy_640000.safetensors'
            cfg=dict(env=dict(voc_scale=0.),embodiment=dict(config={}),action=dict(wrist_kp=25.,wrist_tau_max=45.))
            tensors={'actor':np.arange(12,dtype=np.float32).reshape(3,4)}
            save_file(tensors,str(parent),metadata={'user.resolved_config_json':json.dumps(cfg),'user.environment_steps':'640000'})
            with patch('own_geometry.validate'):
                result=derive(root,parent,dict(kp=500.,kd=28.,angular_kp=125.))
            self.assertEqual(weights_digest(load_file(parent)),weights_digest(load_file(result)))
            with safe_open(result,framework='numpy') as file:metadata=file.metadata()
            self.assertEqual(metadata['user.environment_steps'],'640000');self.assertEqual(metadata['user.control_calibration_new_training_steps'],'0')
            self.assertEqual(json.loads(metadata['user.original_training_resolved_config_json']),cfg)
            actual=json.loads(metadata['user.resolved_config_json'])
            self.assertEqual(actual['action']['wrist_tau_max'],45.)
            self.assertEqual(actual['embodiment']['config']['joint_overrides'][r'wrist_[xyz]$']['effort_limit'],45.)
            self.assertEqual(actual['embodiment']['config']['joint_overrides'][r'wrist_[xyz]$']['kp'],500.)
