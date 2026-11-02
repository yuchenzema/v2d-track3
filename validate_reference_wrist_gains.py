"""Check that reference-wrist trials retain learned finger control."""
from types import SimpleNamespace
import numpy as np
from residual_gain_controller import gains_for_layout,source_timed_gains,scale_action
layout=SimpleNamespace(action_dim=56,hand_action_starts=(0,28),finger_counts=(22,22))
profile={'wrist_position':0.,'wrist_orientation':0.,'finger':1.}
values=gains_for_layout(layout,profile)
assert np.all(values[:6]==0) and np.all(values[6:28]==1)
assert np.all(values[28:34]==0) and np.all(values[34:]==1)
try:gains_for_layout(layout,{**profile,'finger':0.})
except ValueError:pass
else:raise AssertionError('A fully disabled learned actor is not an eligible policy trial')
print('Reference wrist channels and learned finger channels validated; fully disabled actor rejected.')
profile={'wrist_position':.5,'wrist_orientation':1.,'finger':1.,'after_source_frame':40,'ramp_source_frames':10}
g=source_timed_gains(layout,profile,161,161)
assert np.all(g[:41]==1.) and np.allclose(g[45,:3],.75) and np.all(g[50:,:3]==.5)
assert np.all(g[:,3:28]==1.) and np.all(g[:,31:]==1.)
import warp as wp
action=wp.ones(112,dtype=wp.float32,device='cpu');frame=wp.array([0,60],dtype=wp.int32,device='cpu')
output=wp.zeros_like(action);gains=wp.array(g,dtype=wp.float32,device='cpu')
wp.launch(scale_action,dim=112,device='cpu',inputs=[action,gains,frame,161,56,output])
np.testing.assert_array_equal(output.numpy().reshape(2,56),np.stack([g[0],g[60]]))
print('Source-timed gain contract passed: per-world clocks, smooth transition, pickup unchanged, fingers fully learned.')
window=source_timed_gains(layout,{**profile,'until_source_frame':100,'fade_source_frames':10},161,161)
np.testing.assert_array_equal(window[:41],1.)
np.testing.assert_array_equal(window[50:101,:3],.5)
np.testing.assert_allclose(window[105,:3],.75)
np.testing.assert_array_equal(window[110:],1.)
print('Bounded source-phase contract passed: correction fades out before withdrawal without changing pickup or late commands.')
from residual_gain_controller import calibrate_wrist_translation
bias=np.zeros((161,6),np.float32);bias[60]=[.03,0,.025,-.03,0,-.008]
targets=wp.zeros(20,dtype=wp.float32,device='cpu');processed=wp.zeros(32,dtype=wp.float32,device='cpu')
ids=wp.array([0,1,2,5,6,7],dtype=wp.int32,device='cpu');starts=wp.array([0,8],dtype=wp.int32,device='cpu')
wp.launch(calibrate_wrist_translation,dim=(2,2,3),device='cpu',inputs=[frame,wp.array(bias,dtype=wp.float32,device='cpu'),ids,starts,161,2,10,16,targets,processed])
expected=np.zeros((2,10),np.float32);expected[1,[0,1,2,5,6,7]]=bias[60]
np.testing.assert_array_equal(targets.numpy().reshape(2,10),expected)
expected=np.zeros((2,16),np.float32);expected[1,[0,1,2,8,9,10]]=bias[60]
np.testing.assert_array_equal(processed.numpy().reshape(2,16),expected)
print('Declared wrist translation calibration contract passed: only selected robot translation motor targets change.')
