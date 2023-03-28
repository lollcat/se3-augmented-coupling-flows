from typing import Tuple, Optional, Callable

import chex
import distrax
import numpy as np
import jax.numpy as jnp
import haiku as hk

from nets.base import NetsConfig, EGNN
from nets.conditioner_mlp import ConditionerMLP
from flow.bijectors.proj_coupling import ProjSplitCoupling

def make_proj_spline(
        graph_features: chex.Array,
        layer_number: int,
        dim: int,
        n_aug: int,
        swap: bool,
        nets_config: NetsConfig,
        origin_on_coupled_pair: bool = True,
        identity_init: bool = True,
        num_bins: int = 10,
        lower: float = -10.,
        upper: float = 10.,
        ) -> ProjSplitCoupling:
    assert n_aug % 2 == 1
    assert dim in (2, 3)  # Currently just written for 2D and 3D

    base_name = f"layer_{layer_number}_swap{swap}"

    n_heads = dim - 1 if origin_on_coupled_pair else dim
    params_per_dim_per_var_group = (3 * num_bins + 1)
    n_variable_groups = ((n_aug + 1) // 2)
    n_invariant_params = dim * n_variable_groups * params_per_dim_per_var_group

    def bijector_fn(params: chex.Array) -> distrax.RationalQuadraticSpline:
        leading_shape = params.shape[:-2]
        # Flatten last 2 axes.
        params = jnp.reshape(params, (*leading_shape, np.prod(params.shape[-2:])))
        mlp_function = ConditionerMLP(
            name=f"conditionermlp_cond_mlp_" + base_name,
            mlp_units=nets_config.mlp_head_config.mlp_units,
            identity_init=identity_init,
            n_output_params=n_invariant_params,
        )
        params = mlp_function(params)
        # reshape
        params = jnp.reshape(params, (*leading_shape, (n_aug + 1) // 2, dim, params_per_dim_per_var_group))
        bijector = distrax.RationalQuadraticSpline(
            params,
            range_min=lower,
            range_max=upper,
            boundary_slopes='unconstrained',
            min_bin_size=(upper - lower) * 1e-4)
        return bijector

    if nets_config.type == "egnn":
        n_invariant_feat_out = nets_config.egnn_torso_config.n_invariant_feat_hidden
    elif nets_config.type == "e3gnn":
        n_invariant_feat_out = nets_config.e3gnn_torso_config.n_invariant_feat_hidden
    elif nets_config.type == "egnn_v0":
        n_invariant_feat_out = nets_config.egnn_v0_torso_config.n_invariant_feat_hidden
    else:
        raise NotImplementedError

    equivariant_fn = EGNN(name=base_name,
                          nets_config=nets_config,
                          n_equivariant_vectors_out=n_heads,
                          n_invariant_feat_out=n_invariant_feat_out,
                          zero_init_invariant_feat=False)

    return ProjSplitCoupling(
        split_index=(n_aug + 1) // 2,
        event_ndims=3,  # [nodes, n_aug+1, dim]
        get_basis_vectors_and_invariant_vals=equivariant_fn,
        graph_features=graph_features,
        bijector=bijector_fn,
        swap=swap,
        split_axis=-2,
        origin_on_coupled_pair=origin_on_coupled_pair
    )
