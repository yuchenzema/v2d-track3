"""Compare grasp-search conventions with the actual native Warp primitives."""
import json
from pathlib import Path
import numpy as np
from scipy.spatial.transform import Rotation
import warp as wp
from flash_chord.runtime.residual import filter_residual

@wp.kernel
def native_values(value:wp.array(dtype=float),q:wp.array(dtype=wp.quat)):
    value[0]=filter_residual(.2,.01,.05,.2,.3)
    q[0]=wp.quat_rpy(.153,.473,.246)

values=wp.zeros(1,dtype=float,device='cpu');quaternions=wp.zeros(1,dtype=wp.quat,device='cpu')
wp.launch(native_values,dim=1,inputs=[values,quaternions],device='cpu')
error=float((Rotation.from_quat(quaternions.numpy()[0]).inv()*Rotation.from_euler('xyz',[.153,.473,.246])).magnitude())
assert abs(float(values.numpy()[0])-.01)<1e-7
assert error<1e-6
report={'native_physical_filter_value':float(values.numpy()[0]),'expected_filter_value':.01,
        'native_rpy_vs_baked_rotation_error_rad':error,'passed':True}
Path('/episode/research/native-grasp-conventions-validation.json').write_text(json.dumps(report,indent=2))
print(json.dumps(report),flush=True)
