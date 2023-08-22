import chex
import distrax
import jax.numpy as jnp
import haiku as hk

from eacf.nets.make_egnn import EGNN, NetsConfig
from eacf.flow.bijectors.generic_split_coupling import SplitCouplingWithExtra
from eacf.flow.aug_flow_dist import GraphFeatures


def make_conditioner(equivariant_fn, get_scaling_weight_fn):
    def conditioner(x: chex.Array, graph_features: GraphFeatures):
        scaling_weight = get_scaling_weight_fn()
        shift = equivariant_fn(x, graph_features) * scaling_weight
        chex.assert_equal_shape([shift, x])
        # Restrict shift to not change the centre of mass.
        # This means the nice flow preserves the centre-of-mass distribution from the layer before.
        shift = shift - jnp.mean(shift, axis=-3, keepdims=True)
        return shift
    return conditioner


def make_se_equivariant_nice(graph_features,
                             layer_number,
                             dim: int,
                             n_aug: int,
                             swap, nets_config: NetsConfig, identity_init: bool = True):
    """Flow is x + (x - r)*scale where scale is an invariant scalar, and r is equivariant reference point"""
    assert n_aug % 2 == 1

    n_vectors_within_coupling = (n_aug + 1) // 2
    equivariant_fn = EGNN(name=f"layer_{layer_number}_swap{swap}",
                          nets_config=nets_config,
                          n_equivariant_vectors_out=n_vectors_within_coupling,
                          n_invariant_feat_out=0,
                          zero_init_invariant_feat=False)

    # Used to for zero initialisation.
    get_scaling_weight_fn = lambda: hk.get_parameter(
        f"layer_{layer_number}_swap{swap}_scaling_weight",  shape=(), init=jnp.zeros if identity_init else
        hk.initializers.Constant(0.0001), dtype=float)

    def bijector_fn(params):
        shift = params
        return distrax.ScalarAffine(log_scale=jnp.zeros_like(shift), shift=shift)

    conditioner = make_conditioner(equivariant_fn, get_scaling_weight_fn=get_scaling_weight_fn)

    nice_bijector = SplitCouplingWithExtra(
        graph_features=graph_features,
        split_index=(n_aug + 1) // 2,
        event_ndims=3,  # [nodes, n_aug+1, dim]
        conditioner=conditioner,
        bijector=bijector_fn,
        swap=swap,
        split_axis=-2
    )
    return nice_bijector