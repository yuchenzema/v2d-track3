"""Verify actor selection depends only on declared source phase and hand identity."""
from types import SimpleNamespace
import numpy as np
from bimanual_policy_controller import scheduled_weights
l=SimpleNamespace(action_dim=56,sides=['right','left'],hand_action_starts=[0,28],finger_counts=[22,22])
spec={'hand_weights':{'left':.5,'right':0.},'source_schedule':{'left':{'initial_weight':0.,'start_source_frame':96,'ramp_source_frames':12}}}
w=scheduled_weights(l,spec,161,161)
assert w.shape==(161,56) and np.all(w[:97]==0.) and np.all(w[:,0:28]==0.)
assert np.all(w[102,28:]==.25) and np.all(w[108:,28:]==.5)
static=scheduled_weights(l,{'hand_weights':spec['hand_weights']},161,161)
assert np.all(static[:,28:]==.5) and np.all(static[:,:28]==0.)
print('Actor phase schedule contract passed: smooth source-clock weights, stable hand identities and unchanged static composition.')
