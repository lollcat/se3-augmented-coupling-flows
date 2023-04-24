import chex
import distrax
import jax.numpy as jnp


from nets.conditioner_mlp import ConditionerHead
from nets.base import MLPHeadConfig
from flow.bijectors.centre_of_mass_invariant_coupling import CentreOfMassInvariantSplitCoupling


def make_spherical_coupling_layer(
        graph_features: chex.Array,
        layer_number: int,
        dim: int,
        n_aug: int,
        swap: bool,
        mlp_head_config: MLPHeadConfig,
        transformer_config: TransformerConfig,
        identity_init: bool = True,
        spline_num_bins: int = 4,
        spline_max_abs_min: float = 10.,
        use_aux_loss: bool = True,
        n_inner_transforms: int = 1,
        ) -> CentreOfMassInvariantSplitCoupling:
    assert n_aug % 2 == 1
    assert dim in (2, 3)  # Currently just written for 2D and 3D
    base_name = f"layer_{layer_number}_swap{swap}"

    multiplicity_within_coupling_split = ((n_aug + 1) // 2)
    params_per_dim = 3 * spline_num_bins + 1
    n_invariant_params = multiplicity_within_coupling_split * dim * params_per_dim
    n_vectors_per_inner_transform = dim


    def bijector_fn(params: chex.Array, vector_index: int) -> distrax.Bijector:
        chex.assert_rank(params, 2)
        n_nodes, n_dim = params.shape
        # Flatten last 2 axes.
        mlp_function = ConditionerHead(
            name=f"conditionermlp_cond_mlp_vector{vector_index}" + base_name,
            mlp_units=mlp_head_config.mlp_units,
            zero_init=identity_init,
            n_output_params=n_invariant_params,
            stable_layer=mlp_head_config.stable
        )
        params = mlp_function(params)
        # reshape
        params = jnp.reshape(params, (n_nodes, (n_aug + 1) // 2, dim, params_per_dim))
        bijector = distrax.RationalQuadraticSpline(
            params,
            range_min=-spline_max_abs_min,
            range_max=spline_max_abs_min,
            boundary_slopes='unconstrained',
            min_bin_size=(spline_max_abs_min*2) * 1e-4)
        return bijector

    def transformer_forward(positions: chex.Array, features: chex.Array) -> chex.Array:
        chex.assert_rank(positions, 3)
        chex.assert_rank(features, 3)
        n_nodes, n_vec_multiplicity_in, dim = positions.shape
        assert n_vec_multiplicity_in == multiplicity_within_coupling_split
        # net = EGNN(name=base_name,
        #               nets_config=nets_config,
        #               n_equivariant_vectors_out=n_vectors_per_inner_transform*multiplicity_within_coupling_split*n_inner_transforms,
        #               n_invariant_feat_out=n_invariant_feat_out,
        #               zero_init_invariant_feat=False)
        h = net(positions, features)
        h = jnp.reshape(h, (n_nodes, multiplicity_within_coupling_split, n_inner_transforms,
                                        n_vectors_per_inner_transform, dim))
        return h


    return CentreOfMassInvariantSplitCoupling(
        split_index=(n_aug + 1) // 2,
        graph_features=graph_features,
        bijector=bijector_fn,
        swap=swap,
        n_inner_transforms=n_inner_transforms,
        event_ndims=3,  # [nodes, n_aug+1, dim]
        split_axis=-2,
        conditioner=transformer_forward
    )
