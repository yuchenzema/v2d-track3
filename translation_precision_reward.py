"""Training-only position precision, retaining native orientation-sensitive reward.

This changes a scalar learning objective only. Evaluation imports neither this
module nor its installer; states, commands, actions and physics are untouched.
"""
import warp as wp

@wp.kernel
def mix_translation_precision(
 body_q:wp.array(dtype=wp.transform),body_ids:wp.array(dtype=wp.int32),
 target:wp.array(dtype=wp.vec3),num_bodies:int,bodies_per_world:int,
 fraction:float,scale:float,frame_dt:float,term_count:int,
 object_term:wp.array(dtype=wp.float32),terms:wp.array(dtype=wp.float32),
 weights:wp.array(dtype=wp.float32),score:wp.array(dtype=wp.float32)):
 world=wp.tid();precision=float(0.)
 for b in range(num_bodies):
  actual=wp.transform_get_translation(body_q[world*bodies_per_world+body_ids[b]])
  precision+=wp.exp(-wp.length(actual-target[world*num_bodies+b])/scale)
 precision/=float(num_bodies)
 previous=object_term[world];updated=(1.-fraction)*previous+fraction*precision
 object_term[world]=updated;terms[world*term_count]=updated
 score[world]+=frame_dt*weights[0]*(updated-previous)

def install(fraction=.25,scale=.01):
 assert 0.<fraction<=.5 and .005<=scale<=.03
 from flash_chord.objectives.tracking import TrackingObjective
 from flash_chord.objectives.config import OBJECTIVE_TERM_NAMES
 assert OBJECTIVE_TERM_NAMES[0]=='object_keypoints'
 original=TrackingObjective.evaluate
 assert not getattr(original,'_translation_precision',False),'Already installed'
 def evaluate(self,state,timestep,terminated):
  original(self,state,timestep,terminated)
  if self.enabled[0]:
   wp.launch(mix_translation_precision,dim=self.world_count,inputs=[
    state.body_q,self.command.body_ids,self.command.body_target_pos_w,self.num_bodies,
    self.bodies_per_world,float(fraction),float(scale),self.buffers.frame_dt,len(OBJECTIVE_TERM_NAMES),
    self.buffers.object_keypoints,self.buffers.terms,self.buffers.weights,self.buffers.score])
 evaluate._translation_precision=True
 TrackingObjective.evaluate=evaluate

if __name__=='__main__':
 # A multi-world CPU contract checks indexing, unit conversion and score scaling.
 import numpy as np
 wp.init();device='cpu';n=3;B=2;K=11
 poses=np.zeros((n*4,7),np.float32);poses[:,6]=1
 for w,x in enumerate([0.,.01,.04]):
  poses[w*4+1,0]=x;poses[w*4+3,0]=-x
 body=wp.array(poses,dtype=wp.transform,device=device);ids=wp.array([1,3],dtype=wp.int32,device=device)
 target=wp.zeros(n*B,dtype=wp.vec3,device=device);old=np.array([.7,.6,.5],np.float32)
 obj=wp.array(old,device=device);terms=wp.array(np.full(n*K,2.,np.float32),device=device)
 weights=wp.array(np.r_[20.,np.ones(K-1)].astype(np.float32),device=device);score=wp.array(np.array([1.,2.,3.],np.float32),device=device)
 before=body.numpy().copy();before_target=target.numpy().copy()
 wp.launch(mix_translation_precision,dim=n,device=device,inputs=[body,ids,target,B,4,.25,.01,.05,K,obj,terms,weights,score])
 expected=.75*old+.25*np.exp(-np.array([0.,.01,.04])/.01)
 np.testing.assert_allclose(obj.numpy(),expected,rtol=1e-6)
 np.testing.assert_allclose(score.numpy(),[1.,2.,3.]+expected-old,rtol=1e-6)
 np.testing.assert_array_equal(body.numpy(),before);np.testing.assert_array_equal(target.numpy(),before_target)
 np.testing.assert_array_equal(terms.numpy().reshape(n,K)[:,1:],2.)
 print('Translation precision CPU contract passed: multi-world offsets, metres, dt, term isolation, and unchanged physics inputs.')
