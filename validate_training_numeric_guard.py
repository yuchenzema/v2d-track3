"""Invalid physics must count as failure without mutating physical states."""
import numpy as np,warp as wp
from training_numeric_failure_guard import reject_invalid_transition,recover_terminal_buffers,clear_failed_contact_readout,clear_failed_solver_cache
n=3;B=2;nq=4;nd=3;K=11;D=6
body=np.zeros((n*B,7),np.float32);body[:,6]=1;body[B,0]=np.nan
vel=np.zeros((n*B,6),np.float32);vel[2*B,4]=1001.
q=np.zeros(n*nq,np.float32);qd=np.zeros(n*nd,np.float32)
device='cpu';poses=wp.array(body,dtype=wp.transform,device=device);speed=wp.array(vel,dtype=wp.spatial_vector,device=device)
score=wp.array(np.array([.7,np.nan,.8],np.float32),device=device);terms=wp.ones(n*K,dtype=wp.float32,device=device)
weights=np.ones(K,np.float32);weights[6]=-20
done=wp.zeros(n,dtype=wp.int32,device=device);flags=wp.zeros_like(done);counts=wp.zeros_like(done);diagnostics=wp.zeros(n*D,dtype=wp.float32,device=device)
wp.launch(reject_invalid_transition,dim=n,device=device,inputs=[poses,speed,wp.array(q,device=device),wp.array(qd,device=device),B,nq,nd,K,6,.05,D,2,wp.array(weights,device=device),score,terms,done,diagnostics,flags,counts])
np.testing.assert_array_equal(flags.numpy(),[0,1,1]);np.testing.assert_array_equal(done.numpy(),[0,1,1]);np.testing.assert_array_equal(counts.numpy(),[0,1,1])
np.testing.assert_allclose(score.numpy(),[.7,-1.,-1.]);np.testing.assert_array_equal(poses.numpy(),body);np.testing.assert_array_equal(speed.numpy(),vel)
v=terms.numpy().reshape(n,K);assert np.all(v[0]==1) and np.all(v[1:,6]==1) and np.all(np.delete(v[1:],6,axis=1)==0)
current=wp.array(np.arange(12,dtype=np.float32),device=device);terminal=wp.array(np.full(12,np.nan,np.float32),device=device)
wp.launch(recover_terminal_buffers,dim=(n,4),device=device,inputs=[flags,4,current,terminal])
assert np.isnan(terminal.numpy()[:4]).all();np.testing.assert_array_equal(terminal.numpy()[4:],current.numpy()[4:])
print('Numeric guard contract passed: failures counted, physical states unchanged, finite negative rewards, valid worlds untouched, terminal replay buffers recovered from native reset.')
contacts=wp.array(np.ones((n*2,3),np.float32),dtype=wp.vec3,device=device)
wp.launch(clear_failed_contact_readout,dim=(n,2),device=device,inputs=[flags,2,contacts])
np.testing.assert_array_equal(contacts.numpy()[:2],1.);np.testing.assert_array_equal(contacts.numpy()[2:],0.)
cache=wp.array(np.ones((n,4),np.float32),dtype=wp.float32,device=device)
wp.launch(clear_failed_solver_cache,dim=(n,4),device=device,inputs=[flags,cache])
np.testing.assert_array_equal(cache.numpy()[0],1.);np.testing.assert_array_equal(cache.numpy()[1:],0.)
print('Failed-world contact-readout and solver-cache recovery passed; valid worlds unchanged.')
