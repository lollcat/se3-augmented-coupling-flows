import distrax
import jax.nn
import jax.numpy as jnp

from flow.nets import se_equivariant_fn, se_invariant_fn


def make_conditioner(equivariant_fn=se_equivariant_fn, invariant_fn=se_invariant_fn,
                     identity_init: bool = True, mlp_units=(5, 5)):
    def conditioner(x):
        log_scale_param = invariant_fn(x, 1, mlp_units=mlp_units, zero_init=False)*1e-3  # Nan's for 0 init?
        scale = jax.nn.softplus(log_scale_param)
        reference_vector = equivariant_fn(x, mlp_units=mlp_units, zero_init=False)
        equivariant_shift = (equivariant_fn(x, zero_init=identity_init, mlp_units=mlp_units) - x)
        shift = - reference_vector * (scale - 1) + equivariant_shift
        return scale, shift
    return conditioner


def make_se_equivariant_vector_scale_shift(dim, swap, identity_init: bool = True, mlp_units=(5, 5)):
    """Flow is x + (x - r)*scale + shift where shift is invariant scalar, and r is equivariant reference point"""

    def bijector_fn(params):
        scale, shift = params
        return distrax.ScalarAffine(scale=scale, shift=shift)

    conditioner = make_conditioner(identity_init=identity_init, mlp_units=mlp_units)
    return distrax.SplitCoupling(
        split_index=dim,
        event_ndims=2,  # [nodes, dim]
        conditioner=conditioner,
        bijector=bijector_fn,
        swap=swap,
        split_axis=-1
    )
