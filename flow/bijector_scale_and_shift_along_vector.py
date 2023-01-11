import distrax
import jax.nn
import jax.numpy as jnp

from flow.nets import se_invariant_net, se_equivariant_net


def make_conditioner(ref_equivariant_fn, shift_equivariant_fn, invariant_fn):
    def conditioner(x):
        log_scale_param = invariant_fn(x)
        log_scale_param = jnp.broadcast_to(log_scale_param, x.shape)
        scale = jax.nn.softplus(log_scale_param)
        reference_point = ref_equivariant_fn(x)
        equivariant_shift = shift_equivariant_fn(x) - x
        shift = - reference_point * (scale - 1) + equivariant_shift
        return scale, shift
    return conditioner


def make_se_equivariant_vector_scale_shift(layer_number, dim, swap, identity_init: bool = True, mlp_units=(5, 5)):
    """Flow is x + (x - r)*scale + shift where scale is an invariant scalar, and r is equivariant reference point"""

    ref_equivariant_fn = se_equivariant_net(name=f"layer_{layer_number}_ref",
                                        zero_init=False,
                                        mlp_units=mlp_units)

    shift_equivariant_fn = se_equivariant_net(name=f"layer_{layer_number}_ref",
                                            zero_init=identity_init,
                                            mlp_units=mlp_units)

    invariant_fn = se_invariant_net(name=f"layer_{layer_number}",
                                    n_vals=1,
                                    zero_init=identity_init,
                                    mlp_units=mlp_units)

    def bijector_fn(params):
        scale, shift = params
        return distrax.ScalarAffine(scale=scale, shift=shift)

    conditioner = make_conditioner(ref_equivariant_fn, shift_equivariant_fn, invariant_fn)
    return distrax.SplitCoupling(
        split_index=dim,
        event_ndims=2,  # [nodes, dim]
        conditioner=conditioner,
        bijector=bijector_fn,
        swap=swap,
        split_axis=-1
    )
