"""Training-only actor normalization freeze for behavior-preserving finetuning.

Critic normalization and all physical dynamics remain native. Actor affine
parameters remain differentiable; running statistics use the successful anchor.
"""
from functools import wraps


def install():
    from flash_chord.training.flash_sac.network import UnitBatchNorm
    original=UnitBatchNorm.__call__
    assert not getattr(original,'_frozen_actor_normalization',False)
    @wraps(original)
    def frozen(self,value,*,training):
        return original(self,value,training=False)
    frozen._frozen_actor_normalization=True
    UnitBatchNorm.__call__=frozen


if __name__=='__main__':
    import jax,jax.numpy as jnp,numpy as np
    from flash_chord.training.flash_sac.network import UnitBatchNorm,EnsembleUnitBatchNorm
    install();layer=UnitBatchNorm();x=jnp.arange(24,dtype=jnp.float32).reshape(4,6)
    variables=layer.init(jax.random.PRNGKey(1),x,training=False)
    trained,mutable=layer.apply(variables,x+10,training=True,mutable=['batch_stats'])
    expected=layer.apply(variables,x+10,training=False)
    np.testing.assert_array_equal(np.asarray(trained),np.asarray(expected))
    for a,b in zip(jax.tree.leaves(variables['batch_stats']),jax.tree.leaves(mutable['batch_stats'])):np.testing.assert_array_equal(np.asarray(a),np.asarray(b))
    gradient=jax.grad(lambda p:jnp.square(layer.apply({'params':p,'batch_stats':variables['batch_stats']},x,training=True)).mean())(variables['params'])
    assert any(np.any(np.asarray(v)!=0) for v in jax.tree.leaves(gradient))
    critic=EnsembleUnitBatchNorm(ensemble_size=2);cx=jnp.stack([x,x+2]);cv=critic.init(jax.random.PRNGKey(2),cx,training=False)
    _,changed=critic.apply(cv,cx+10,training=True,mutable=['batch_stats'])
    assert any(not np.array_equal(np.asarray(a),np.asarray(b)) for a,b in zip(jax.tree.leaves(cv['batch_stats']),jax.tree.leaves(changed['batch_stats'])))
    print('Actor normalization freeze contract passed: fixed running statistics, matching inference, trainable parameters, native critic updates.')
