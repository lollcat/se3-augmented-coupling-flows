from typing import Tuple, Optional, Callable

import chex
import distrax
import jax.numpy as jnp

from nets.base import NetsConfig, build_egnn_fn
from flow.bijectors.pairwise_difference_coupling import SubtractSplitCoupling


def make_conditioner_and_bijector_fn(equivariant_fn, lower, upper):
    def conditioner(x: chex.Array, graph_features: chex.Array) -> chex.Array:
        x, _ = equivariant_fn(x, graph_features)
        params = jnp.swapaxes(x, -1, -2)  # Move the (3 * num_bins + 1) axis to be last.
        return params

    def bijector_fn(params: chex.Array):
        bijector = distrax.RationalQuadraticSpline(
            params,
            range_min=lower,
            range_max=upper,
            boundary_slopes='unconstrained',
            min_bin_size=(upper - lower) * 1e-4)
        return bijector

    return conditioner, bijector_fn


def make_equivariant_fn(nets_config, layer_number, swap, n_heads):
    n_invariant_feat_out = 1  # Not used.

    equivariant_fn = build_egnn_fn(name=f"layer_{layer_number}_swap{swap}",
                                   nets_config=nets_config,
                                   n_equivariant_vectors_out=n_heads,
                                   n_invariant_feat_out=n_invariant_feat_out,
                                   zero_init_invariant_feat=False)
    return equivariant_fn

def make_equi_spline(
        graph_features: chex.Array,
        layer_number: int,
        dim: int,
        n_aug: int,
        swap: bool,
        nets_config: NetsConfig,
        num_bins: int = 4,
        lower: float = -10.,
        upper: float = 10.,
        identity_init: bool = True,
        ) -> SubtractSplitCoupling:
    assert n_aug % 2 == 1
    assert dim in (2, 3)  # Currently just written for 2D and 3D

    params_per_dim_per_var_group = (3 * num_bins + 1)
    equivariant_fn = make_equivariant_fn(nets_config=nets_config,
                                         layer_number=layer_number, swap=swap,
                                         n_heads=params_per_dim_per_var_group)
    conditioner, bijector_fn = make_conditioner_and_bijector_fn(equivariant_fn, lower=lower, upper=upper)

    return SubtractSplitCoupling(
        split_index=(n_aug + 1) // 2,
        event_ndims=3,  # [nodes, n_aug+1, dim]
        conditioner=conditioner,
        bijector=bijector_fn,
        graph_features = graph_features,
        swap=swap,
        split_axis=-2
    )
