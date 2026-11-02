"""Check mapped robot residual scaling without simulation or learned-policy claims."""
import numpy as np,warp as wp
from wrist_impedance_controller import rescale_residual,gains_for_layout
from types import SimpleNamespace
layout=SimpleNamespace(action_dim=18,sides=['right','left'],hand_action_starts=[0,9])
g=gains_for_layout(layout,{'right':{'kp':100.,'kd':12.,'original_kp':25.}})
assert np.all(g[:3]==.25) and np.all(g[3:]==1.)
def mapped(a,s,code):
 if code==0:return a
 if code==1:return s*np.clip(a,-1,1)
 a=np.clip(a,-1,1);return a/((1-abs(a))+abs(a)/s)
for code in range(3):
 a=np.tile(np.linspace(-.95,.95,18,dtype=np.float32),3);s=np.linspace(.25,4.,18,dtype=np.float32);out=wp.empty_like(wp.array(a,device='cpu'))
 wp.launch(rescale_residual,dim=len(a),inputs=[wp.array(a,device='cpu'),wp.array(g,device='cpu'),wp.array(s,device='cpu'),code,18],outputs=[out],device='cpu')
 value=out.numpy();assert np.allclose(mapped(value,np.tile(s,3),code),mapped(a,np.tile(s,3),code)*np.tile(g,3),atol=2e-6)
 assert np.array_equal(value[np.tile(g,3)==1],a[np.tile(g,3)==1])
print('Impedance residual contract passed for raw, clipped-linear and nonlinear normalized input mappings; only selected wrist translations change.')
