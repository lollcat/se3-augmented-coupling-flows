import distrax
import jax.nn
import jax.numpy as jnp

from flow.nets import se_equivariant_net


def make_conditioner(ref_and_scale_equivariant_fn, shift_equivariant_fn, activation_fn = jax.nn.softplus):
    def conditioner(x):
        reference_point, log_scale_param = ref_and_scale_equivariant_fn(x)
        log_scale_param = jnp.broadcast_to(log_scale_param, x.shape)
        scale = activation_fn(log_scale_param)
        equivariant_shift = shift_equivariant_fn(x) - x
        shift = - reference_point * (scale - 1) + equivariant_shift
        return scale, shift
    return conditioner


def make_se_equivariant_vector_scale_shift(layer_number, dim, swap, identity_init: bool = True, mlp_units=(5, 5)):
    """Flow is x + (x - r)*scale + shift where scale is an invariant scalar, and r is equivariant reference point"""

    ref_and_scale_equivariant_fn = se_equivariant_net(name=f"layer_{layer_number}_ref",
                                        zero_init=False,
                                        mlp_units=mlp_units, h_out=True, h_out_dim=1)

    shift_equivariant_fn = se_equivariant_net(name=f"layer_{layer_number}_shift",
                                            zero_init=identity_init,
                                            mlp_units=mlp_units, h_out=False)


    def bijector_fn(params):
        scale, shift = params
        return distrax.ScalarAffine(scale=scale, shift=shift)

    conditioner = make_conditioner(ref_and_scale_equivariant_fn, shift_equivariant_fn)
    return distrax.SplitCoupling(
        split_index=dim,
        event_ndims=2,  # [nodes, dim]
        conditioner=conditioner,
        bijector=bijector_fn,
        swap=swap,
        split_axis=-1
    )
