"""Exercise grip correction arithmetic on Warp CPU before physical trials."""
from types import SimpleNamespace
import numpy as np
import warp as wp
from grip_bias_controller import apply_bias,offset_table

names=('index_MCP_FE','index_MCP_AA','index_PIP','index_DIP','thumb_IP')
layout=SimpleNamespace(sides=('right','left'),hand_action_starts=(0,11),finger_counts=(5,5),action_dim=22)
hands={side:SimpleNamespace(finger_joint_names=tuple(side+'_'+n for n in names)) for side in layout.sides}
table=offset_table(17,9,[{'side':'left','start':4,'end':5}],layout,hands,[{'left':.1,'right':0.}],ramp=2.)
assert not table[:4].any() and not table[-1].any()
assert not table[:,:,:17].any() and not table[:,:,18].any()
action=np.linspace(-.8,.8,22,dtype=np.float32)
scale=np.full(22,.15,np.float32);s=np.full(22,1./(.7*.15),np.float32)
for mapping in [0,1,2]:
    out=wp.zeros(22,dtype=wp.float32,device='cpu')
    wp.launch(apply_bias,dim=22,inputs=[wp.array(action,device='cpu'),wp.array([8],dtype=wp.int32,device='cpu'),
        wp.array(table,device='cpu'),wp.array(scale,device='cpu'),wp.array(s,device='cpu'),mapping,17,22],outputs=[out],device='cpu')
    actual=out.numpy();bias=table[8,0]
    assert np.array_equal(actual[bias==0],action[bias==0])
    def mapped(a):
        if mapping==0:return a
        if mapping==1:return np.clip(a,-1,1)*s
        return a/((1-np.abs(a))+np.abs(a)/s)
    assert np.allclose((mapped(actual)-mapped(action))*scale,bias,atol=2e-7)
print('Warp CPU: physical flexion increments match declared radians; wrists, abduction and zero-profile actions remain bit-identical.')
