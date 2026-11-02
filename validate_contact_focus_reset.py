"""Verify contact-focused training resets stay bounded, masked and unassisted."""
import numpy as np,warp as wp
from training_finger_reset import MixedFingerReferenceReset
from flash_chord.lifecycle.reset import ReferenceResetConfig
n=1024;r=MixedFingerReferenceReset.build(n,344,ReferenceResetConfig(always_reset_to_first_frame=True,reset_finger_openness=1.,voc_decay_steps=0,seed=46),device='cpu')
r.configure_focus(167,267,1.);mask=wp.ones(n,dtype=wp.int32,device='cpu');voc=wp.zeros(1,dtype=wp.float32,device='cpu');r.sample(mask,voc)
x=r.reset_frame.numpy().copy();f=r.finger_scale.numpy();assert x.min()>=167 and x.max()<=267 and len(np.unique(x))>90
assert np.all(r.applied_voc_scale.numpy()==0.) and 400<np.sum(f==1.)<624
r.focus_probability.assign(np.array([0.],np.float32));selected=(np.arange(n)%2)==0;mask.assign(selected.astype(np.int32));r.sample(mask,voc)
y=r.reset_frame.numpy();assert np.array_equal(y[~selected],x[~selected]);assert np.all(y[selected]==0)
assert np.all(r.applied_voc_scale.numpy()==0.);print('Contact focus reset contract passed: bounded source interval, masked updates, exact-reference coverage, adjustable captured buffer and zero object support.')
r.set_focus_interval(110,160);r.focus_probability.assign(np.array([1.],np.float32));mask.fill_(1);r.sample(mask,voc)
z=r.reset_frame.numpy();assert z.min()>=110 and z.max()<=160 and len(np.unique(z))>40
assert np.all(r.applied_voc_scale.numpy()==0.)
print('Dynamic focus interval contract passed: range changes through persistent buffers; object support stays zero.')
