import distrax
import jax.numpy as jnp

from nets.nets import se_equivariant_net, EgnnConfig


def make_conditioner(equivariant_fn):
    def conditioner(x):
        shift = equivariant_fn(x) - x
        return shift
    return conditioner


def make_se_equivariant_nice(layer_number, dim, swap, egnn_config: EgnnConfig, identity_init: bool = True):
    """Flow is x + (x - r)*scale where scale is an invariant scalar, and r is equivariant reference point"""

    ref_equivariant_fn = se_equivariant_net(
        egnn_config._replace(name=f"layer_{layer_number}_swap{swap}_ref",
                           identity_init_x=identity_init,
                           h_config=egnn_config.h_config._replace(h_out=False)))

    def bijector_fn(params):
        shift = params
        return distrax.ScalarAffine(log_scale=jnp.zeros_like(shift), shift=shift)

    conditioner = make_conditioner(ref_equivariant_fn)
    return distrax.SplitCoupling(
        split_index=dim,
        event_ndims=2,  # [nodes, dim]
        conditioner=conditioner,
        bijector=bijector_fn,
        swap=swap,
        split_axis=-1
    )
