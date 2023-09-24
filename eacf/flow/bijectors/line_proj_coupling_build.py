from typing import Tuple

import chex
import distrax
import jax.numpy as jnp
import jax

from eacf.nets.make_egnn import NetsConfig, EGNN
from eacf.nets.conditioner_mlp import ConditionerHead
from eacf.flow.bijectors.line_proj_coupling import LineSplitCoupling
from eacf.utils.numerical import inverse_softplus


def make_line_proj_coupling_layer(
        graph_features: chex.Array,
        layer_number: int,
        dim: int,
        n_aug: int,
        swap: bool,
        nets_config: NetsConfig,
        identity_init: bool = True,
        origin_on_coupled_pair: bool = False,
        transform_type: str = 'real_nvp',
        num_bins: int = 10,
        lower: float = -10.,
        upper: float = 10.,
        n_inner_transforms: int = 1
        ) -> LineSplitCoupling:
    assert n_aug % 2 == 1
    assert dim in (2, 3)  # Currently just written for 2D and 3D
    base_name = f"layer_{layer_number}_swap{swap}"

    multiplicity_within_coupling_split = ((n_aug + 1) // 2)
    n_heads = 1 if origin_on_coupled_pair else 2
    if transform_type == "real_nvp":
        params_per_channel = 2
        n_invariant_params_bijector = params_per_channel*multiplicity_within_coupling_split
    elif transform_type == 'spline':
        params_per_channel = (3 * num_bins + 1)
        n_invariant_params_bijector = multiplicity_within_coupling_split * params_per_channel
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
            params = jnp.reshape(params, (n_nodes, (n_aug + 1) // 2, 2))
            log_scale, shift = jnp.split(params, axis=-1, indices_or_sections=2)

            # If log_scale is initialised to 0 then this initialises the flow to the identity.
            log_scale = log_scale + inverse_softplus(jnp.array(1.))
            scale = jax.nn.softplus(log_scale)

            return distrax.ScalarAffine(scale=scale, shift=shift)
        else:
            params = jnp.reshape(params,
                                 (n_nodes, multiplicity_within_coupling_split, 1, params_per_channel))
            bijector = distrax.RationalQuadraticSpline(
                params,
                range_min=lower,
                range_max=upper,
                boundary_slopes='unconstrained',
                min_bin_size=(upper - lower) * 1e-4)
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
                      n_equivariant_vectors_out=n_heads*multiplicity_within_coupling_split*n_inner_transforms,
                      n_invariant_feat_out=n_invariant_feat_out,
                      zero_init_invariant_feat=False)
        vectors, h = net(positions, features)
        vectors = jnp.reshape(vectors, (n_nodes, multiplicity_within_coupling_split, n_inner_transforms, n_heads, dim))
        return vectors, h


    return LineSplitCoupling(
        split_index=(n_aug + 1) // 2,
        origin_on_coupled_pair=origin_on_coupled_pair,
        get_basis_vectors_and_invariant_vals=equivariant_fn,
        graph_features=graph_features,
        bijector=bijector_fn,
        swap=swap,
        n_inner_transforms=n_inner_transforms,
        event_ndims=3,  # [nodes, n_aug+1, dim]
        split_axis=-2,
    )

