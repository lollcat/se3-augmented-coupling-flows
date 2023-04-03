from typing import Tuple

import chex
import distrax
import numpy as np
import jax.numpy as jnp
import haiku as hk
import tensorflow_probability.substrates.jax as tfp

from nets.base import NetsConfig, EGNN
from nets.conditioner_mlp import ConditionerMLP
from flow.bijectors.spherical_coupling import SphericalSplitCoupling
from flow.bijectors.blockwise import Blockwise


def make_spherical_coupling_layer(
        graph_features: chex.Array,
        layer_number: int,
        dim: int,
        n_aug: int,
        swap: bool,
        nets_config: NetsConfig,
        identity_init: bool = True,
        n_transforms: int = 1,
        spline_num_bins: int = 4,
        spline_max: float = 10.
        ) -> SphericalSplitCoupling:
    assert n_aug % 2 == 1
    assert dim in (2, 3)  # Currently just written for 2D and 3D
    base_name = f"layer_{layer_number}_swap{swap}"

    multiplicity_within_coupling_split = ((n_aug + 1) // 2)
    params_per_dim = 3 * spline_num_bins + 1
    n_invariant_params = multiplicity_within_coupling_split * dim * params_per_dim
    n_vectors = n_transforms * dim


    def bijector_fn(params: chex.Array, vector_index: int) -> distrax.Bijector:
        chex.assert_rank(params, 2)
        n_nodes, n_dim = params.shape
        # Flatten last 2 axes.
        mlp_function = ConditionerMLP(
            name=f"conditionermlp_cond_mlp_vector{vector_index}" + base_name,
            mlp_units=nets_config.mlp_head_config.mlp_units,
            identity_init=identity_init,
            n_output_params=n_invariant_params,
                                      )
        params = mlp_function(params)
        # reshape
        params = jnp.reshape(params, (n_nodes, (n_aug + 1) // 2, dim, params_per_dim))
        d_bijector = distrax.RationalQuadraticSpline(
            params[:, :, :1, :],
            range_min=0.0,
            range_max=spline_max,
            boundary_slopes='lower_identity',
            min_bin_size=(spline_max - 0.0) * 1e-4)
        if dim == 2:
            theta_bijector = distrax.RationalQuadraticSpline(
                params[:, :, 1:2, :],
                range_min=-jnp.pi,
                range_max=jnp.pi,
                boundary_slopes='circular',
                min_bin_size=(spline_max - 0.0) * 1e-4)
            bijector = Blockwise(
                bijectors=[d_bijector, theta_bijector],
                split_indices=[1, ],
            )
        else:
            assert dim == 3
            theta_bijector = distrax.RationalQuadraticSpline(
                params[:, :, 1:2, :],
                range_min=0,
                range_max=jnp.pi,
                boundary_slopes='circular',
                min_bin_size=(spline_max - 0.0) * 1e-4)
            torsional_bijector = distrax.RationalQuadraticSpline(
                params[:, :, 2:3, :],
                range_min=-jnp.pi,
                range_max=jnp.pi,
                boundary_slopes='circular',
                min_bin_size=(spline_max - 0.0) * 1e-4)
            bijector = Blockwise(
                bijectors=[d_bijector, theta_bijector, torsional_bijector],
                split_indices=[1,2],
            )
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
                      n_equivariant_vectors_out=n_vectors*multiplicity_within_coupling_split,
                      n_invariant_feat_out=n_invariant_feat_out,
                      zero_init_invariant_feat=False)
        vectors, h = net(positions, features)
        vectors = jnp.reshape(vectors, (n_nodes, multiplicity_within_coupling_split, n_vectors, dim))
        return vectors, h


    return SphericalSplitCoupling(
        split_index=(n_aug + 1) // 2,
        event_ndims=3,  # [nodes, n_aug+1, dim]
        get_reference_vectors_and_invariant_vals=equivariant_fn,
        graph_features=graph_features,
        bijector=bijector_fn,
        swap=swap,
        split_axis=-2,
    )