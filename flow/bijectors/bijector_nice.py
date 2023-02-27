import distrax
import jax.numpy as jnp
import haiku as hk

from nets.base import build_egnn_fn, NetsConfig


def make_conditioner(equivariant_fn, get_scaling_weight_fn):
    def conditioner(x):
        shift = (equivariant_fn(x) - x) * get_scaling_weight_fn()
        return shift
    return conditioner


def make_se_equivariant_nice(layer_number, dim, swap, nets_config: NetsConfig, identity_init: bool = True):
    """Flow is x + (x - r)*scale where scale is an invariant scalar, and r is equivariant reference point"""

    equivariant_fn = build_egnn_fn(name=f"layer_{layer_number}_swap{swap}",
                                   nets_config=nets_config,
                                   n_equivariant_vectors_out=1,
                                   n_invariant_feat_out=0,
                                   zero_init_invariant_feat=False)

    # Used to for zero initialisation.
    get_scaling_weight_fn = lambda: hk.get_parameter(
        f"layer_{layer_number}_swap{swap}_scaling_weight",  shape=(), init=jnp.zeros if identity_init else jnp.ones)

    def bijector_fn(params):
        shift = params
        return distrax.ScalarAffine(log_scale=jnp.zeros_like(shift), shift=shift)

    conditioner = make_conditioner(equivariant_fn, get_scaling_weight_fn=get_scaling_weight_fn)
    return distrax.SplitCoupling(
        split_index=dim,
        event_ndims=2,  # [nodes, dim]
        conditioner=conditioner,
        bijector=bijector_fn,
        swap=swap,
        split_axis=-1
    )
