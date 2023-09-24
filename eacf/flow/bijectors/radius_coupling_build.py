from typing import Tuple

import chex
import distrax
import jax.numpy as jnp

from eacf.nets.make_egnn import NetsConfig, EGNN
from eacf.nets.conditioner_mlp import ConditionerHead
from eacf.flow.bijectors.radius_coupling import RadialSplitCoupling


def make_radial_coupling_layer(
        graph_features: chex.Array,
        layer_number: int,
        dim: int,
        n_aug: int,
        swap: bool,
        nets_config: NetsConfig,
        identity_init: bool = True,
        spline_num_bins: int = 4,
        dist_spline_max: float = 10.,
        use_aux_loss: bool = True,
        n_inner_transforms: int = 1,
        ) -> RadialSplitCoupling:
    assert n_aug % 2 == 1
    assert dim in (2, 3)  # Currently just written for 2D and 3D
    base_name = f"layer_{layer_number}_swap{swap}"

    multiplicity_within_coupling_split = ((n_aug + 1) // 2)
    params_per_dim = 3 * spline_num_bins + 1
    n_invariant_params = multiplicity_within_coupling_split * params_per_dim


    def bijector_fn(params: chex.Array, vector_index: int) -> distrax.Bijector:
        chex.assert_rank(params, 2)
        n_nodes, n_dim = params.shape
        # Flatten last 2 axes.
        mlp_function = ConditionerHead(
            name=f"conditionermlp_cond_mlp_vector{vector_index}" + base_name,
            mlp_units=nets_config.mlp_head_config.mlp_units,
            zero_init=identity_init,
            n_output_params=n_invariant_params,
            stable_layer=nets_config.mlp_head_config.stable
        )
        params = mlp_function(params)
        # reshape
        params = jnp.reshape(params, (n_nodes, (n_aug + 1) // 2, 1, params_per_dim))
        bijector = distrax.RationalQuadraticSpline(
            params,
            range_min=0.0,
            range_max=dist_spline_max,
            boundary_slopes='unconstrained',
            min_bin_size=(dist_spline_max - 0.0) * 1e-4)
        return bijector


    if nets_config.type == "egnn":
        n_invariant_feat_out = nets_config.egnn_torso_config.n_invariant_feat_hidden
    else:
        raise NotImplementedError

    def equivariant_fn(positions: chex.Array, features: chex.Array) -> Tuple[chex.Array, chex.Array]:
        chex.assert_rank(positions, 3)
        chex.assert_rank(features, 3)
        n_nodes, n_vec_multiplicity_in, dim = positions.shape
        assert n_vec_multiplicity_in == multiplicity_within_coupling_split
        net = EGNN(name=base_name,
                      nets_config=nets_config,
                      n_equivariant_vectors_out=multiplicity_within_coupling_split*n_inner_transforms,
                      n_invariant_feat_out=n_invariant_feat_out,
                      zero_init_invariant_feat=False)
        vectors, h = net(positions, features)
        vectors = jnp.reshape(vectors, (n_nodes, multiplicity_within_coupling_split, n_inner_transforms, dim))
        return vectors, h


    return RadialSplitCoupling(
        split_index=(n_aug + 1) // 2,
        get_reference_vectors_and_invariant_vals=equivariant_fn,
        graph_features=graph_features,
        bijector=bijector_fn,
        swap=swap,
        use_aux_loss=use_aux_loss,
        n_inner_transforms=n_inner_transforms,
        event_ndims=3,  # [nodes, n_aug+1, dim]
        split_axis=-2,
    )
