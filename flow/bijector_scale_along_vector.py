import distrax
import jax.nn
import jax.numpy as jnp

from flow.nets import se_equivariant_net, EgnnConfig


def make_conditioner(ref_and_scale_equivariant_fn, activation_fn):
    def conditioner(x):
        reference_point, log_scale_param = ref_and_scale_equivariant_fn(x)
        if activation_fn == jax.nn.softplus:
            inverse_softplus = lambda x: jnp.log(jnp.exp(x) - 1.)
            log_scale_param = log_scale_param + inverse_softplus(jnp.array(1.0))
        log_scale_param = jnp.broadcast_to(log_scale_param, x.shape)
        scale = activation_fn(log_scale_param)
        shift = - reference_point * (scale - 1)
        return scale, shift
    return conditioner


def make_se_equivariant_scale_along_vector(layer_number, dim, swap, egnn_config: EgnnConfig, identity_init: bool = True,
                                           activation_fn = jax.nn.softplus):
    """Flow is x + (x - r)*scale where scale is an invariant scalar, and r is equivariant reference point"""

    ref_and_scale_equivariant_fn = se_equivariant_net(
        egnn_config._replace(name=f"layer_{layer_number}_ref",
                           identity_init_x=False,
                           zero_init_h=identity_init,
                           h_config=egnn_config.h_config._replace(h_out_dim=1, h_out=True)))

    def bijector_fn(params):
        scale, shift = params
        return distrax.ScalarAffine(scale=scale, shift=shift)

    conditioner = make_conditioner(ref_and_scale_equivariant_fn, activation_fn=activation_fn)
    return distrax.SplitCoupling(
        split_index=dim,
        event_ndims=2,  # [nodes, dim]
        conditioner=conditioner,
        bijector=bijector_fn,
        swap=swap,
        split_axis=-1
    )
