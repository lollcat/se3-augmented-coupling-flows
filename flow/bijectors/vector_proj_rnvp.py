import chex
import distrax
import numpy as np
import jax.numpy as jnp
import haiku as hk

from nets.base import NetsConfig, EGNN
from nets.conditioner_mlp import ConditionerMLP
from flow.bijectors.vector_proj_coupling import VectorProjSplitCoupling




def make_vector_proj_realnvp(
        graph_features: chex.Array,
        layer_number: int,
        dim: int,
        n_aug: int,
        swap: bool,
        nets_config: NetsConfig,
        identity_init: bool = True,
        add_small_identity: bool = True,
        n_vectors: int = 1,
        transform_type = 'real_nvp',
        num_bins: int = 4,
        lower: float = -4.,
        upper: float = 4.,
        ) -> VectorProjSplitCoupling:
    assert n_aug % 2 == 1
    assert dim in (2, 3)  # Currently just written for 2D and 3D
    base_name = f"layer_{layer_number}_swap{swap}"


    n_invariant_params = ((n_aug + 1) // 2)
    if transform_type == 'real_nvp':
        params_per_dim = 2
    elif transform_type == 'spline':
        params_per_dim = (3 * num_bins + 1)
    else:
        raise NotImplementedError
    n_invariant_params = n_invariant_params * params_per_dim


    def bijector_fn(params: chex.Array, vector_index: int) -> distrax.Bijector:
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
            return distrax.ScalarAffine(log_scale=log_scale, shift=shift)
        elif transform_type == 'spline':
            params = params[:, :, None, :]
            return distrax.RationalQuadraticSpline(params, lower, upper)
        else:
            raise NotImplementedError


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
                          n_equivariant_vectors_out=n_vectors,
                          n_invariant_feat_out=n_invariant_feat_out,
                          zero_init_invariant_feat=False)

    return VectorProjSplitCoupling(
        split_index=(n_aug + 1) // 2,
        event_ndims=3,  # [nodes, n_aug+1, dim]
        get_reference_vectors_and_invariant_vals=equivariant_fn,
        graph_features=graph_features,
        bijector=bijector_fn,
        swap=swap,
        split_axis=-2,
        add_small_identity=add_small_identity
    )
