"""Keep the observed thumb-index pinch while opening the three free fingers."""
import numpy as np
from release_controller import selected_finger_columns,apply_release
import warp as wp
names=tuple(f'right_{finger}_joint_{j}' for finger in ['thumb','index','middle','ring','pinky'] for j in range(2))
selected=selected_finger_columns(names,{'finger_joint_prefixes':['right_middle_','right_ring_','right_pinky_']})
np.testing.assert_array_equal(selected,np.arange(4,10))
worlds=2;frames=3;fingers=20;dofs=24;actiondim=32;processdim=34
weights=np.zeros((frames,fingers),np.float32);weights[:,selected]=np.array([0,.5,1.])[:,None]
weights=wp.array(weights,dtype=wp.float32,device='cpu')
targets=wp.ones(worlds*dofs,dtype=wp.float32,device='cpu');processed=wp.ones(worlds*processdim,dtype=wp.float32,device='cpu');filtered=wp.ones(worlds*actiondim,dtype=wp.float32,device='cpu')
wp.launch(apply_release,dim=(worlds,fingers),device='cpu',inputs=[wp.array([1,2],dtype=wp.int32,device='cpu'),weights,
 wp.zeros(frames*fingers,dtype=wp.float32,device='cpu'),wp.array(np.arange(fingers),dtype=wp.int32,device='cpu'),
 wp.array(np.arange(fingers),dtype=wp.int32,device='cpu'),wp.array(np.arange(fingers),dtype=wp.int32,device='cpu'),
 frames,fingers,dofs,processdim,actiondim,targets,processed,filtered])
for a,n in [(targets,dofs),(processed,processdim),(filtered,actiondim)]:
 x=a.numpy().reshape(worlds,n);np.testing.assert_array_equal(x[:,:4],1);np.testing.assert_array_equal(x[:,10:],1)
 np.testing.assert_array_equal(x[0,4:10],.5);np.testing.assert_array_equal(x[1,4:10],0.)
print('Selective release contract passed: thumb/index and opposite hand unchanged; selected finger commands and filtered actions open at each world clock.')
