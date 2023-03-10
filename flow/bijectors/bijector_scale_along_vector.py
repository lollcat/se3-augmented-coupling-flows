import distrax
import jax.nn
import jax.numpy as jnp

from nets.base import NetsConfig, build_egnn_fn


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


def make_se_equivariant_scale_along_vector(layer_number, dim, swap, nets_config: NetsConfig, identity_init: bool = True,
                                           activation_fn = jax.nn.softplus):
    """Flow is x + (x - r)*scale where scale is an invariant scalar, and r is equivariant reference point"""

    equivariant_fn = build_egnn_fn(name=f"layer_{layer_number}_swap{swap}",
                                   nets_config=nets_config,
                                   n_equivariant_vectors_out=1,
                                   n_invariant_feat_out=1,
                                   zero_init_invariant_feat=identity_init)

    def bijector_fn(params):
        scale, shift = params
        return distrax.ScalarAffine(scale=scale, shift=shift)

    conditioner = make_conditioner(equivariant_fn, activation_fn=activation_fn)
    return distrax.SplitCoupling(
        split_index=dim,
        event_ndims=2,  # [nodes, dim]
        conditioner=conditioner,
        bijector=bijector_fn,
        swap=swap,
        split_axis=-1
    )
