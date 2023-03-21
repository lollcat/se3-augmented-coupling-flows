from typing import Tuple, Optional, Callable

import chex
import distrax
import numpy as np
import jax.numpy as jnp
import haiku as hk

from nets.base import NetsConfig, build_egnn_fn
from flow.bijectors.proj import ProjSplitCoupling

def make_se_equivariant_split_coupling_with_projection(
        graph_features: chex.Array,
        layer_number: int,
        dim: int,
        n_aug: int,
        swap: bool,
        nets_config: NetsConfig,
        identity_init: bool = True,
        ) -> ProjSplitCoupling:
    assert n_aug % 2 == 1
    assert dim in (2, 3)  # Currently just written for 2D and 3D

    n_heads = dim
    n_invariant_params = dim * 2 * ((n_aug + 1) // 2)

    mlp_function = hk.Sequential(
        name=f"layer_{layer_number}_swap{swap}_cond_mlp",
        layers=[
            hk.LayerNorm(axis=-1, create_offset=True, create_scale=True, param_axis=-1),
            hk.nets.MLP(nets_config.mlp_head_config.mlp_units, activate_final=True),
            hk.Linear(n_invariant_params, b_init=jnp.zeros, w_init=jnp.zeros) if identity_init else
            hk.Linear(n_invariant_params,
                      b_init=hk.initializers.VarianceScaling(0.01),
                      w_init=hk.initializers.VarianceScaling(0.01))
        ])

    def bijector_fn(params: chex.Array) -> distrax.ScalarAffine:
        leading_shape = params.shape[:-2]
        # Flatten last 2 axes.
        params = jnp.reshape(params, (*leading_shape, np.prod(params.shape[-2:])))
        params = mlp_function(params)
        # reshape
        params = jnp.reshape(params, (*leading_shape, (n_aug + 1) // 2, dim*2))
        log_scale, shift = jnp.split(params, axis=-1, indices_or_sections=2)
        return distrax.ScalarAffine(log_scale=log_scale, shift=shift)



    n_coupling_variable_groups = n_aug + 1
    if nets_config.type == "mace":
        n_invariant_feat_out = int(nets_config.mace_torso_config.n_invariant_feat_residual
                                       / (n_coupling_variable_groups / 2))
    elif nets_config.type == "egnn":
        n_invariant_feat_out = int(nets_config.egnn_torso_config.h_embedding_dim
                                       / (n_coupling_variable_groups / 2))
    elif nets_config.type == 'e3transformer':
        n_invariant_feat_out = int(nets_config.e3transformer_lay_config.n_invariant_feat_hidden
                                       / (n_coupling_variable_groups / 2))
    elif nets_config.type == "e3gnn":
        n_invariant_feat_out = int(nets_config.e3gnn_torso_config.n_invariant_feat_hidden
                                       / (n_coupling_variable_groups / 2))
    else:
        raise NotImplementedError

    equivariant_fn = build_egnn_fn(name=f"layer_{layer_number}_swap{swap}",
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
        split_axis=-2
    )
