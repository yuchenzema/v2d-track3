"""Check reset initialization coverage and zero support without running a physics scene."""
import numpy as np
import warp as wp
from flash_chord.lifecycle.reset import ReferenceResetConfig
from training_finger_reset import MixedFingerReferenceReset

count=1024
reset=MixedFingerReferenceReset.build(count,37,ReferenceResetConfig(
    always_reset_to_first_frame=True,reset_finger_openness=1.,voc_decay_steps=0,seed=44),device='cpu')
mask=wp.ones(count,dtype=wp.int32,device='cpu');voc=wp.zeros(1,dtype=wp.float32,device='cpu')
reset.sample(mask,voc)
first=reset.finger_scale.numpy().copy()
assert np.all(reset.reset_frame.numpy()==0)
assert np.all(reset.applied_voc_scale.numpy()==0.)
assert np.all((first>=0.)&(first<=1.))
exact=int(np.sum(first==1.));assert 400<=exact<=624
assert np.min(first)<.01 and np.sum(first<.5)>100
selected=np.arange(count)%2==0;mask.assign(selected.astype(np.int32));reset.sample(mask,voc)
assert np.array_equal(first[~selected],reset.finger_scale.numpy()[~selected])
print('Reset contract verified:',exact,'exact-reference worlds, remaining worlds cover open-to-closed poses; zero object support; unselected worlds unchanged.',flush=True)

# The adaptive curriculum counts only completed, exact-initialization episodes
# that began at the current boundary. It never changes object state or actions.
from types import SimpleNamespace
reset.configure_adaptive(minimum_frame=10,opening_gate_end=20,stride=2)
reset.environment=SimpleNamespace(
    done=wp.zeros(count,dtype=wp.int32,device='cpu'),
    truncation=wp.ones(count,dtype=wp.int32,device='cpu'),
    timestep=wp.full(count,19,dtype=wp.int32,device='cpu'),
    episode_start_frame=wp.full(count,10,dtype=wp.int32,device='cpu'))
assert reset.maybe_advance(0) is None
mask.fill_(1)
reset.finger_scale.fill_(1.)
reset.sample(mask,voc)
assert np.array_equal(reset.opening_counts.numpy(),[count,count])
starts=reset.reset_frame.numpy()
assert set(np.unique(starts))=={0,6,10}
assert 600<np.sum(starts==10)<800 and 50<np.sum(starts==0)<160
decision=reset.maybe_advance(100_352)
assert decision['advanced'] and decision['minimum_reference_frame']==8
reset.environment.episode_start_frame.fill_(8)
reset.environment.timestep.fill_(12)
reset.environment.done.fill_(1)
reset.environment.truncation.zero_()
reset.finger_scale.fill_(1.)
reset.sample(mask,voc)
decision=reset.maybe_advance(200_704)
assert not decision['advanced'] and decision['opening_success_fraction_in_training']==0.
assert set(np.unique(reset.reset_frame.numpy()))=={0,4,8}
assert np.all(reset.applied_voc_scale.numpy()==0.)
print('Adaptive reset contract verified: advances only after successful exact-initialization episodes; failures retain boundary; zero support.',flush=True)
