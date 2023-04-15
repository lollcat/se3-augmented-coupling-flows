from typing import Tuple

import chex
import distrax
import jax.numpy as jnp

from nets.base import NetsConfig, EGNN
from nets.conditioner_mlp import ConditionerHead
from flow.bijectors.proj_coupling import ProjSplitCoupling


_N_ADDITIONAL_BASIS_VECTORS_LOEWDIM = 1

def make_proj_coupling_layer(
        graph_features: chex.Array,
        layer_number: int,
        dim: int,
        n_aug: int,
        swap: bool,
        nets_config: NetsConfig,
        identity_init: bool = True,
        origin_on_coupled_pair: bool = False,
        add_small_identity: bool = True,
        orthogonalization_method: str = 'gram-schmidt',  # or loewdin
        transform_type: str = 'real_nvp',
        num_bins: int = 10,
        lower: float = -10.,
        upper: float = 10.,
        n_inner_transforms: int = 1
        ) -> ProjSplitCoupling:
    assert n_aug % 2 == 1
    assert dim in (2, 3)  # Currently just written for 2D and 3D
    base_name = f"layer_{layer_number}_swap{swap}"

    multiplicity_within_coupling_split = ((n_aug + 1) // 2)
    n_heads = dim - 1 if origin_on_coupled_pair else dim
    n_heads = n_heads + _N_ADDITIONAL_BASIS_VECTORS_LOEWDIM if orthogonalization_method == 'loewdin' else n_heads
    if transform_type == "real_nvp":
        params_per_dim_channel = dim * 2
        n_invariant_params_bijector = params_per_dim_channel*multiplicity_within_coupling_split
    elif transform_type == 'spline':
        params_per_dim_per_var_group = (3 * num_bins + 1)
        params_dim_channel = params_per_dim_per_var_group * dim
        n_invariant_params_bijector = multiplicity_within_coupling_split * params_dim_channel
    else:
        raise NotImplementedError

    def bijector_fn(params: chex.Array, transform_index: int) -> distrax.Bijector:
        chex.assert_rank(params, 2)
        n_nodes, n_dim = params.shape
        mlp_function = ConditionerHead(
            name=f"conditionermlp_cond_mlp_{transform_index}" + base_name,
            mlp_units=nets_config.mlp_head_config.mlp_units,
            zero_init=identity_init,
            n_output_params=n_invariant_params_bijector,
            stable_layer=nets_config.mlp_head_config.stable

        )
        params = mlp_function(params)
        if transform_type == 'real_nvp':
            # reshape
            params = jnp.reshape(params, (n_nodes, (n_aug + 1) // 2, dim*2))
            log_scale, shift = jnp.split(params, axis=-1, indices_or_sections=2)
            return distrax.ScalarAffine(log_scale=log_scale, shift=shift)
        else:
            params = jnp.reshape(params,
                                 (n_nodes, multiplicity_within_coupling_split, dim, params_per_dim_per_var_group))
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
    else:
        raise NotImplementedError

    def equivariant_fn(positions: chex.Array, features: chex.Array) -> Tuple[chex.Array, chex.Array]:
        chex.assert_rank(positions, 3)
        chex.assert_rank(features, 3)
        n_nodes, n_vec_multiplicity_in, dim = positions.shape
        assert n_vec_multiplicity_in == multiplicity_within_coupling_split
        net = EGNN(name=base_name,
                      nets_config=nets_config,
                      n_equivariant_vectors_out=n_heads*multiplicity_within_coupling_split*n_inner_transforms,
                      n_invariant_feat_out=n_invariant_feat_out,
                      zero_init_invariant_feat=False)
        vectors, h = net(positions, features)
        vectors = jnp.reshape(vectors, (n_nodes, multiplicity_within_coupling_split, n_inner_transforms, n_heads, dim))
        return vectors, h


    return ProjSplitCoupling(
        split_index=(n_aug + 1) // 2,
        event_ndims=3,  # [nodes, n_aug+1, dim]
        origin_on_coupled_pair=origin_on_coupled_pair,
        get_basis_vectors_and_invariant_vals=equivariant_fn,
        graph_features=graph_features,
        bijector=bijector_fn,
        swap=swap,
        split_axis=-2,
        add_small_identity=add_small_identity,
        orthogonalization_method=orthogonalization_method,
        n_inner_transforms=n_inner_transforms
    )
