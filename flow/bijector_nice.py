import distrax
import jax.numpy as jnp

from flow.nets import se_equivariant_fn, se_invariant_fn


def make_conditioner(equivariant_fn=se_equivariant_fn, identity_init: bool = True, mlp_units=(5, 5)):
    def conditioner(x):
        shift = equivariant_fn(x, zero_init=identity_init, mlp_units=mlp_units) - x
        return shift
    return conditioner


def make_se_equivariant_nice(dim, swap, identity_init: bool = True, mlp_units=(5, 5)):
    def bijector_fn(shift):
        return distrax.ScalarAffine(log_scale=jnp.zeros_like(shift), shift=shift)

    conditioner = make_conditioner(identity_init=identity_init, mlp_units=mlp_units)
    return distrax.SplitCoupling(
        split_index=dim,
        event_ndims=2,  # [nodes, dim]
        conditioner=conditioner,
        bijector=bijector_fn,
        swap=swap,
        split_axis=-1
    )
