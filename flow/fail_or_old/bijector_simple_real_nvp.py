import distrax
import jax.numpy as jnp

from nets import equivariant_fn, invariant_fn


def make_conditioner(equivariant_fn=equivariant_fn, invariant_fn=invariant_fn):
    def conditioner(x):
        # Get scale and shift, initialise to be small.
        log_scale = invariant_fn(x, 1, zero_init=False)
        shift = equivariant_fn(x, zero_init=False) - x
        return jnp.broadcast_to(log_scale, shift.shape), shift
    return conditioner


def make_se_equivariant_split_coupling_simple(dim, swap):
    def bijector_fn(params):
        log_scale, shift = params
        return distrax.ScalarAffine(log_scale=log_scale, shift=shift)

    conditioner = make_conditioner()
    return distrax.SplitCoupling(
        split_index=dim,
        event_ndims=2,  # [nodes, dim]
        conditioner=conditioner,
        bijector=bijector_fn,
        swap=swap,
        split_axis=-1
    )
