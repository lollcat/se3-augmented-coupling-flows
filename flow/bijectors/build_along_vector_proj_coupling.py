from typing import Tuple

import chex
import distrax
import numpy as np
import jax.numpy as jnp
import haiku as hk
import tensorflow_probability.substrates.jax as tfp

from nets.base import NetsConfig, EGNN
from nets.conditioner_mlp import ConditionerMLP
from flow.bijectors.along_vector_proj_coupling import VectorProjSplitCoupling


def make_vector_proj(
        graph_features: chex.Array,
        layer_number: int,
        dim: int,
        n_aug: int,
        swap: bool,
        nets_config: NetsConfig,
        identity_init: bool = True,
        n_vectors: int = 1,
        global_frame: bool = True,
        transform_type: str = 'real_nvp',
        spline_num_bins: int = 4,
        spline_max: float = 10.
        ) -> VectorProjSplitCoupling:
    assert n_aug % 2 == 1
    assert dim in (2, 3)  # Currently just written for 2D and 3D
    base_name = f"layer_{layer_number}_swap{swap}"

    multiplicity_within_coupling_split = ((n_aug + 1) // 2)
    if transform_type == 'real_nvp':
        params_per_dim = 2
    elif transform_type == 'spline':
        params_per_dim = 3 * spline_num_bins + 1
    else:
        raise NotImplementedError
    n_invariant_params = multiplicity_within_coupling_split * params_per_dim


    def bijector_fn(params: chex.Array, vector_index: int) -> distrax.Bijector:
        if global_frame and (vector_index == n_vectors):
            max_spline_bin = spline_max*3
        else:
            max_spline_bin = spline_max
        leading_shape = params.shape[:-2]
        # Flatten last 2 axes.
        params = jnp.reshape(params, (*leading_shape, np.prod(params.shape[-2:])))
        mlp_function = ConditionerMLP(
            name=f"conditionermlp_cond_mlp_vector{vector_index}" + base_name,
            mlp_units=nets_config.mlp_head_config.mlp_units,
            identity_init=identity_init,
            n_output_params=n_invariant_params,
                                      )
        params = mlp_function(params)
        # reshape
        params = jnp.reshape(params, (*leading_shape, (n_aug + 1) // 2, params_per_dim))
        if transform_type == 'real_nvp':
            log_scale, shift = jnp.split(params, axis=-1, indices_or_sections=2)
            chex.assert_shape(log_scale, (*leading_shape, (n_aug + 1) // 2, 1))
            inner_bijector = distrax.ScalarAffine(log_scale=log_scale, shift=shift)
            bijector = distrax.Chain([tfp.bijectors.Exp(), inner_bijector, distrax.Inverse(tfp.bijectors.Exp())])
            return bijector
        elif transform_type == "spline":
            params = jnp.reshape(params, (*leading_shape, (n_aug + 1) // 2, 1, params_per_dim))
            bijector = distrax.RationalQuadraticSpline(
                params,
                range_min=0.0,
                range_max=max_spline_bin,
                boundary_slopes='lower_identity',
                min_bin_size=(spline_max - 0.0) * 1e-4)
            return bijector
        else:
            raise NotImplementedError


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
        if global_frame:
            global_mean = jnp.mean(positions, axis=0, keepdims=True)
            global_vector = global_mean - positions
            vectors = jnp.concatenate([vectors, global_vector[:, :, None, :]], axis=-2)
        return vectors, h

    return VectorProjSplitCoupling(
        split_index=(n_aug + 1) // 2,
        event_ndims=3,  # [nodes, n_aug+1, dim]
        get_reference_vectors_and_invariant_vals=equivariant_fn,
        graph_features=graph_features,
        bijector=bijector_fn,
        swap=swap,
        split_axis=-2,
    )