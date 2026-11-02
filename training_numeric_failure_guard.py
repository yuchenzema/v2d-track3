"""Treat invalid simulator transitions as failed training episodes before replay.

Only training with native auto-reset is supported. Physical states and forces
are not edited by this guard. The existing RL reset performs episode recovery;
canonical evaluation never imports this module.
"""
import numpy as np,warp as wp

@wp.kernel
def reject_invalid_transition(body:wp.array(dtype=wp.transform),velocity:wp.array(dtype=wp.spatial_vector),
 q:wp.array(dtype=wp.float32),qd:wp.array(dtype=wp.float32),bodies:int,numq:int,numdof:int,
 term_count:int,termination_index:int,frame_dt:float,diagnostic_count:int,cause_count:int,
 weights:wp.array(dtype=wp.float32),score:wp.array(dtype=wp.float32),terms:wp.array(dtype=wp.float32),
 terminated:wp.array(dtype=wp.int32),diagnostics:wp.array(dtype=wp.float32),
 invalid:wp.array(dtype=wp.int32),counts:wp.array(dtype=wp.int32)):
 world=wp.tid();bad=int(0)
 for b in range(bodies):
  transform=body[world*bodies+b];p=wp.transform_get_translation(transform);r=wp.transform_get_rotation(transform)
  for a in range(3):
   if not wp.isfinite(p[a]) or wp.abs(p[a])>1000.:bad=1
  norm=float(0.)
  for a in range(4):
   if not wp.isfinite(r[a]):bad=1
   norm+=r[a]*r[a]
  if not wp.isfinite(norm) or norm<.5 or norm>1.5:bad=1
  for a in range(6):
   v=velocity[world*bodies+b][a]
   if not wp.isfinite(v) or wp.abs(v)>1000.:bad=1
 for a in range(numq):
  v=q[world*numq+a]
  if not wp.isfinite(v) or wp.abs(v)>1000.:bad=1
 for a in range(numdof):
  v=qd[world*numdof+a]
  if not wp.isfinite(v) or wp.abs(v)>1000.:bad=1
 if not wp.isfinite(score[world]):bad=1
 invalid[world]=bad
 if bad!=0:
  counts[world]+=1;terminated[world]=1
  for k in range(term_count):terms[world*term_count+k]=0.
  terms[world*term_count+termination_index]=1.
  score[world]=frame_dt*weights[termination_index]
  # Undefined diagnostics are failure sentinels, never successful zeros.
  for k in range(diagnostic_count):diagnostics[world*diagnostic_count+k]=1.

@wp.kernel
def recover_terminal_buffers(invalid:wp.array(dtype=wp.int32),dimension:int,current:wp.array(dtype=wp.float32),
 terminal:wp.array(dtype=wp.float32)):
 world,index=wp.tid()
 if invalid[world]!=0:terminal[world*dimension+index]=current[world*dimension+index]

@wp.kernel
def clear_failed_contact_readout(invalid:wp.array(dtype=wp.int32),slots:int,values:wp.array(dtype=wp.vec3)):
 world,index=wp.tid()
 if invalid[world]!=0:values[world*slots+index]=wp.vec3(0.)

@wp.kernel
def clear_failed_solver_cache(invalid:wp.array(dtype=wp.int32),values:wp.array2d(dtype=wp.float32)):
 world,index=wp.tid()
 if invalid[world]!=0:values[world,index]=0.

def install():
 from flash_chord.envs.rl import RLEnv
 from flash_chord.objectives.tracking import TrackingObjective
 from flash_chord.objectives.config import OBJECTIVE_TERM_NAMES
 assert not getattr(RLEnv,'_numeric_training_guard',False)
 native_init=RLEnv.__init__;native_objective=TrackingObjective.evaluate;native_step=RLEnv._rl_step
 termination_index=OBJECTIVE_TERM_NAMES.index('termination')
 def initialize(self,*args,**kwargs):
  native_init(self,*args,**kwargs)
  assert self.config.auto_reset,'Numerical recovery is restricted to training auto-reset environments'
  assert isinstance(self.objective,TrackingObjective) and float(self.objective.weights.numpy()[termination_index])<0
  self._numeric_invalid=wp.zeros(self.world_count,dtype=wp.int32,device=self.device)
  self._numeric_failure_counts=wp.zeros(self.world_count,dtype=wp.int32,device=self.device)
  self.objective._numeric_training_env=self
  self._numeric_solver_caches=[]
  data=getattr(self.solver,'mjw_data',None)
  if data is not None:
   for name in ['qacc','qacc_warmstart']:
    value=getattr(data,name,None)
    if value is not None and value.ndim==2 and value.shape[0]==self.world_count:self._numeric_solver_caches.append(value)
 def objective(self,state,timestep,terminated):
  native_objective(self,state,timestep,terminated);env=getattr(self,'_numeric_training_env',None)
  if env is None:return
  wp.launch(reject_invalid_transition,dim=self.world_count,inputs=[state.body_q,state.body_qd,state.joint_q,state.joint_qd,
   self.bodies_per_world,env.model.joint_coord_count//env.world_count,env.num_joint_dof,len(OBJECTIVE_TERM_NAMES),termination_index,
   self.buffers.frame_dt,len(env.termination.packed_diagnostic_names),len(env.termination.cause_names),self.weights,self.score,self.terms,
   terminated,env.termination.packed_diagnostics,env._numeric_invalid,env._numeric_failure_counts],device=env.device)
  if env.contact is not None:
   slots=env.contact.layout.slots_per_world
   for name in ['contact_pos_w','contact_force_w','contact_pos_b','contact_force_direction_b','contact_pos_o','contact_force_direction_o']:
    wp.launch(clear_failed_contact_readout,dim=(env.world_count,slots),
     inputs=[env._numeric_invalid,slots,getattr(env.contact,name)],device=env.device)
 def step(self):
  native_step(self)
  for cache in self._numeric_solver_caches:
   wp.launch(clear_failed_solver_cache,dim=cache.shape,inputs=[self._numeric_invalid,cache],device=self.device)
  if self.config.capture_terminal_state:
   wp.launch(recover_terminal_buffers,dim=(self.world_count,self.observation_dim),
    inputs=[self._numeric_invalid,self.observation_dim,self.observation,self.terminal_observation],device=self.device)
   context=self.critic_context.size//self.world_count
   if context:wp.launch(recover_terminal_buffers,dim=(self.world_count,context),
    inputs=[self._numeric_invalid,context,self.critic_context,self.terminal_critic_context],device=self.device)
 RLEnv.__init__=initialize;TrackingObjective.evaluate=objective;RLEnv._rl_step=step;RLEnv._numeric_training_guard=True
