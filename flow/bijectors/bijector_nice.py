import distrax
import jax.numpy as jnp
import haiku as hk

from nets.base import build_egnn_fn, NetsConfig
from flow.distrax_with_extra import SplitCouplingWithExtra


def make_conditioner(equivariant_fn, get_scaling_weight_fn, graph_features):
    def conditioner(x):
        shift = equivariant_fn(x, graph_features) * get_scaling_weight_fn()
        shift = jnp.squeeze(shift, axis=-2)  # Only want 1 vector per input vector
        return shift
    return conditioner

def make_se_equivariant_nice(graph_features,
                             layer_number,
                             dim: int,
                             n_aux: int,
                             swap, nets_config: NetsConfig, identity_init: bool = True):
    """Flow is x + (x - r)*scale where scale is an invariant scalar, and r is equivariant reference point"""
    assert n_aux % 2 == 1

    equivariant_fn = build_egnn_fn(name=f"layer_{layer_number}_swap{swap}",
                                   nets_config=nets_config,
                                   n_equivariant_vectors_out=1,
                                   n_invariant_feat_out=0,
                                   zero_init_invariant_feat=False)

    # Used to for zero initialisation.
    get_scaling_weight_fn = lambda: hk.get_parameter(
        f"layer_{layer_number}_swap{swap}_scaling_weight",  shape=(), init=jnp.zeros if identity_init else
        hk.initializers.Constant(0.0001))

    def bijector_fn(params):
        shift = params
        return distrax.ScalarAffine(log_scale=jnp.zeros_like(shift), shift=shift)

    conditioner = make_conditioner(equivariant_fn, get_scaling_weight_fn=get_scaling_weight_fn,
                                   graph_features=graph_features)
    return SplitCouplingWithExtra(
        split_index=(n_aux + 1)//2,
        event_ndims=3,  # [n_var_groups, nodes, dim]
        conditioner=conditioner,
        bijector=bijector_fn,
        swap=swap,
        split_axis=-2
    )
