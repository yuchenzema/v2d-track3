"""Check equivalent actuator force, frame scheduling and isolated wrist channels."""
import numpy as np,warp as wp
from timed_wrist_impedance_controller import impedance_targets,tables
profile={'right':{'kp':250,'kd':20,'start_source_frame':84,'ramp_source_frames':16}}
kp,kd=tables(profile,['left','right'],173,173);assert np.all(kp[:85]==25) and np.all(kp[100,3:]==250) and np.all(kp[:,:3]==25)
assert np.all(kp[92,3:]==137.5) and np.all(kd[92,3:]==13.)
vertical={'left':{'kp':250,'kd':18,'start_source_frame':60,'ramp_source_frames':8,'until_source_frame':95,'fade_source_frames':12,'axes':['z']}}
a,b=tables(vertical,['left','right'],161,161)
assert np.all(a[:61]==25) and a[68,2]==250 and a[95,2]==250 and np.all(a[107:]==25)
assert np.all(a[:,[0,1,3,4,5]]==25) and np.all(b[:,[0,1,3,4,5]]==6)
n=2;nq=10;nd=8;F=173;H=2
q=np.linspace(-.2,.2,n*nq,dtype=np.float32);qd=np.linspace(-.1,.1,n*nd,dtype=np.float32);target=np.linspace(-.3,.3,n*nd,dtype=np.float32)
ref=np.full((F*H,3),.1,np.float32);qi=np.array([0,1,2,5,6,7],np.int32);di=np.array([0,1,2,4,5,6],np.int32);frames=np.array([0,120],np.int32)
out=wp.array(target,device='cpu');wp.launch(impedance_targets,dim=(n,H,3),device='cpu',inputs=[wp.array(frames,dtype=wp.int32,device='cpu'),wp.array(kp,dtype=wp.float32,device='cpu'),wp.array(kd,dtype=wp.float32,device='cpu'),wp.array(ref,dtype=wp.vec3,device='cpu'),wp.array(q,device='cpu'),wp.array(qd,device='cpu'),wp.array(qi,dtype=wp.int32,device='cpu'),wp.array(di,dtype=wp.int32,device='cpu'),wp.array(target,device='cpu'),F,H,nq,nd,out])
actual=out.numpy()
for w in range(n):
 for h in range(H):
  for a in range(3):
   d=w*nd+di[h*3+a];x=q[w*nq+qi[h*3+a]];v=qd[d];r=.1
   native_force=25*(actual[d]-x)-6*v
   expected=kp[frames[w],h*3+a]*(r-x)+25*(target[d]-r)-kd[frames[w],h*3+a]*v
   np.testing.assert_allclose(native_force,expected,atol=2e-5)
np.testing.assert_array_equal(actual.reshape(n,nd)[:,[3,7]],target.reshape(n,nd)[:,[3,7]])
print('Timed impedance contract passed: native actuator force equivalence, independent world clocks, smooth phase schedule, and unchanged non-wrist channels.')
