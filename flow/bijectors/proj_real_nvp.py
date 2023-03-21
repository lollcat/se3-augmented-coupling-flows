from typing import Tuple, Optional, Callable

import chex
import jax
import jax.numpy as jnp
import haiku as hk

from nets.transformer import Transformer
from nets.base import NetsConfig, build_egnn_fn
from utils.numerical import gram_schmidt_fn, rotate_2d, vector_rejection, safe_norm
from flow.distrax_with_extra import SplitCouplingWithExtra, BijectorWithExtra, Array, Extra


def affine_transform_in_new_space(point, change_of_basis_matrix, origin, scale, shift):
    """Perform affine transformation in the space define by the `origin` and `change_of_basis_matrix`, and then
    go back into the original space."""
    chex.assert_rank(point, 1)
    chex.assert_equal_shape((point, scale, shift))
    point_in_new_space = change_of_basis_matrix.T @ (point - origin)
    transformed_point_in_new_space = point_in_new_space * scale + shift
    new_point_original_space = change_of_basis_matrix @ transformed_point_in_new_space + origin
    return new_point_original_space

def inverse_affine_transform_in_new_space(point, change_of_basis_matrix, origin, scale, shift):
    """Inverse of `affine_transform_in_new_space`."""
    chex.assert_rank(point, 1)
    point_in_new_space = change_of_basis_matrix.T  @ (point - origin)
    transformed_point_in_new_space = (point_in_new_space - shift) / scale
    new_point_original_space = change_of_basis_matrix @ transformed_point_in_new_space + origin
    return new_point_original_space


class ProjectedScalarAffine(BijectorWithExtra):
    """Following style of `ScalarAffine` distrax Bijector.

    Note: Doesn't need to operate on batches, as it gets called with vmap."""
    def __init__(self, change_of_basis_matrix, origin, log_scale, shift, info, activation=jax.nn.softplus):
        super().__init__(event_ndims_in=1, is_constant_jacobian=True)
        self._info = info
        self._change_of_basis_matrix = change_of_basis_matrix
        self._origin = origin
        self._shift = shift
        if activation == jnp.exp:
            self._scale = jnp.exp(log_scale)
            self._inv_scale = jnp.exp(jnp.negative(log_scale))
            self._log_scale = log_scale
        else:
            assert activation == jax.nn.softplus
            inverse_softplus = lambda x: jnp.log(jnp.exp(x) - 1.)
            log_scale_param = log_scale + inverse_softplus(jnp.array(1.0))
            self._scale = jax.nn.softplus(log_scale_param)
            self._inv_scale = 1. / self._scale
            self._log_scale = jnp.log(jnp.abs(self._scale))

    def forward(self, x: chex.Array) -> chex.Array:
        """Computes y = f(x)."""
        if len(x.shape) == 3:
            # vmap over n_nodes, n_var_group
            y = jax.vmap(jax.vmap(affine_transform_in_new_space))(x, self._change_of_basis_matrix, self._origin, self._scale,
                                                       self._shift)
        elif len(x.shape) == 4:
            y = jax.vmap(jax.vmap(jax.vmap(affine_transform_in_new_space)))(x, self._change_of_basis_matrix, self._origin, self._scale,
                                                           self._shift)
        else:
            raise Exception
        chex.assert_equal_shape((x, y))
        return y

    def forward_log_det_jacobian(self, x: chex.Array) -> chex.Array:
        """Computes log|det J(f)(x)|."""
        return jnp.sum(self._log_scale, axis=-1)

    def forward_and_log_det(self, x: chex.Array) -> Tuple[chex.Array, chex.Array]:
        """Computes y = f(x) and log|det J(f)(x)|."""
        return self.forward(x), self.forward_log_det_jacobian(x)

    def inverse(self, y: chex.Array) -> chex.Array:
        """Computes x = f^{-1}(y)."""
        if len(y.shape) == 3:
            # vmap over n_nodes, n_var_group
            x = jax.vmap(jax.vmap(inverse_affine_transform_in_new_space))(y, self._change_of_basis_matrix, self._origin, self._scale,
                                                    self._shift)
        elif len(y.shape) == 4:
            # vmap over batch_size, n_nodes, n_var_group
            x = jax.vmap(jax.vmap(jax.vmap(inverse_affine_transform_in_new_space)))(
                y, self._change_of_basis_matrix, self._origin, self._scale, self._shift)
        else:
            raise Exception
        chex.assert_equal_shape((x, y))
        return x

    def inverse_log_det_jacobian(self, y: chex.Array) -> chex.Array:
        """Computes log|det J(f^{-1})(y)|."""
        return jnp.sum(jnp.negative(self._log_scale), -1)

    def inverse_and_log_det(self, y: chex.Array) -> Tuple[chex.Array, chex.Array]:
        """Computes x = f^{-1}(y) and log|det J(f^{-1})(y)|."""
        return self.inverse(y), self.inverse_log_det_jacobian(y)

    def forward_and_log_det_with_extra(self, x: Array) -> Tuple[Array, Array, Extra]:
        y, log_det = self.forward_and_log_det(x)
        return y, log_det, self.get_extra()

    def inverse_and_log_det_with_extra(self, y: Array) -> Tuple[Array, Array, Extra]:
        x, log_det = self.inverse_and_log_det(y)
        return x, log_det, self.get_extra()

    def get_vector_info_single(self, various_x_points):
        basis_vectors = various_x_points[:, 1:]
        basis_vectors = basis_vectors + jnp.eye(basis_vectors.shape[-1])[:basis_vectors.shape[1]][None, :, :]*1e-30
        vec1 = basis_vectors[:, 0]
        vec2 = basis_vectors[:, 1]
        arccos_in = jax.vmap(jnp.dot)(vec1, vec2) / safe_norm(vec1, axis=-1) / safe_norm(vec2, axis=-1)
        theta = jax.vmap(jnp.arccos)(arccos_in)
        log_barrier_in = 1 - jnp.abs(arccos_in) + 1e-6
        aux_loss = - jnp.log(log_barrier_in)
        return theta, aux_loss, log_barrier_in

    def get_extra(self) -> Extra:
        info = {}
        info_aggregator = {}
        various_x_points = self._info['various_x_points']
        if various_x_points.shape[-1] == 3:
            if len(various_x_points.shape) == 5:
                theta, aux_loss, log_barrier_in = jax.vmap(jax.vmap(self.get_vector_info_single))(various_x_points)
            else:
                assert len(various_x_points.shape) == 4
                theta, aux_loss, log_barrier_in = jax.vmap(self.get_vector_info_single)(various_x_points)
            info_aggregator.update(
                mean_abs_theta=jnp.mean, min_abs_theta=jnp.min,
                min_log_barrier_in=jnp.min
            )
            info.update(
                mean_abs_theta=jnp.mean(jnp.abs(theta)), min_abs_theta=jnp.min(jnp.abs(theta)),
                min_log_barrier_in=jnp.min(log_barrier_in)
            )
            aux_loss = jnp.mean(aux_loss)
        else:
            aux_loss = jnp.array(0.)
        info.update(
            shift_norm=jnp.linalg.norm(self._shift, axis=-1)
        )
        extra = Extra(aux_loss=aux_loss, aux_info=info, info_aggregator=info_aggregator)
        if info['shift_norm'].shape != ():
            extra = extra._replace(aux_info=extra.aggregate_info())
        extra = extra._replace(aux_info=jax.lax.stop_gradient(extra.aux_info))
        return extra


def get_new_space_basis(x: chex.Array, various_x_vectors: chex.Array, gram_schmidt: bool, global_frame: bool,
                        add_small_identity: bool = False):

    n_nodes, dim = x.shape

    # Calculate new basis for the affine transform
    various_x_vectors = jnp.swapaxes(various_x_vectors, 0, 1)

    origin = x + various_x_vectors[0]
    basis_vectors = various_x_vectors[1:]

    if add_small_identity:
        # Add independant vectors to try help improve numerical stability
        basis_vectors = basis_vectors + jnp.eye(x.shape[-1])[:basis_vectors.shape[0]][:, None, :]*1e-6

    if global_frame:
        basis_vectors = jnp.mean(basis_vectors, axis=1, keepdims=True)
        origin = jnp.mean(origin, axis=0, keepdims=True)

    if gram_schmidt:
        basis_vectors = jax.tree_map(lambda x: jnp.squeeze(x, axis=0), jnp.split(basis_vectors, dim-1, axis=0))
        assert len(basis_vectors) == (dim - 1)

        orthonormal_vectors = jax.vmap(gram_schmidt_fn)(basis_vectors)
        change_of_basis_matrix = jnp.stack(orthonormal_vectors, axis=-1)
    else:
        chex.assert_tree_shape_suffix(various_x_vectors, (dim, n_nodes, dim))

        z_basis_vector = basis_vectors[0]
        if dim == 3:
            chex.assert_tree_shape_suffix(x, (3,))
            x_basis_vector = basis_vectors[1]
            # Compute reference axes.
            x_basis_vector = vector_rejection(x_basis_vector, z_basis_vector)
            y_basis_vector = jnp.cross(z_basis_vector, x_basis_vector)
            change_of_basis_matrix = jnp.stack([z_basis_vector, x_basis_vector, y_basis_vector], axis=-1)

        else:
            chex.assert_tree_shape_suffix(x, (2,))
            y_basis_vector = rotate_2d(z_basis_vector, theta=jnp.pi * 0.5)
            change_of_basis_matrix = jnp.stack([z_basis_vector, y_basis_vector], axis=-1)

        change_of_basis_matrix = change_of_basis_matrix / safe_norm(change_of_basis_matrix, axis=-2,
                                                                          keepdims=True)


    if global_frame:
        origin = jnp.squeeze(origin, axis=0)
        change_of_basis_matrix = jnp.squeeze(change_of_basis_matrix, axis=0)
        chex.assert_equal_shape((origin, x[0]))
        chex.assert_shape(change_of_basis_matrix, (dim, dim))
    else:
        chex.assert_equal_shape((origin, x))
        chex.assert_shape(change_of_basis_matrix, (n_nodes, dim, dim))
    return origin, change_of_basis_matrix


def make_conditioner(
        graph_features: chex.Array,
        global_frame: bool,
        process_flow_params_jointly: bool,
        multi_x_equivariant_fn: Callable,
        permutation_equivariant_fn: Optional[Callable] = None,
        mlp_function: Optional[Callable] = None,
        gram_schmidt: bool = False,
        condition_on_x_proj: bool = False,
        add_small_identity: bool = False,
                     ):
    if process_flow_params_jointly:
        assert permutation_equivariant_fn is not None
    else:
        assert mlp_function is not None


    def eq_net_out_to_params(x, various_x_points, h):
        chex.assert_rank(x, 2)
        n_nodes, dim = x.shape

        origin, change_of_basis_matrix = get_new_space_basis(x, various_x_points, gram_schmidt, global_frame,
                                                             add_small_identity=add_small_identity)

        if global_frame:
            inv_change_of_basis = change_of_basis_matrix.T  # jnp.linalg.inv(change_of_basis_matrix)
            if condition_on_x_proj:
                x_proj = jax.vmap(lambda x, inv_change_of_basis, origin: inv_change_of_basis @ (x - origin),
                                  in_axes=(0, None, None))(x, inv_change_of_basis, origin)
                bijector_feat_in = jnp.concatenate([x_proj, h], axis=-1)
            else:
                bijector_feat_in = h
            if process_flow_params_jointly:
                log_scale_and_shift = permutation_equivariant_fn(bijector_feat_in)
            else:
                log_scale_and_shift = mlp_function(bijector_feat_in)
            origin = jnp.repeat(origin[None, ...], n_nodes, axis=0)
            change_of_basis_matrix = jnp.repeat(change_of_basis_matrix[None, ...], n_nodes, axis=0)
        else:
            inv_change_of_basis = jax.vmap(lambda x: x.T)(change_of_basis_matrix)
            if process_flow_params_jointly:
                if condition_on_x_proj:
                    x_proj = jax.vmap(jax.vmap(lambda x, inv_change_of_basis, origin:  inv_change_of_basis @ (x - origin),
                                      in_axes=(0, None, None)), in_axes=(None, 0, 0))(
                        x, inv_change_of_basis, origin)
                    bijector_feat_in = jnp.concatenate([x_proj, jnp.repeat(h[None, ...], n_nodes, axis=0)], axis=-1)
                    log_scale_and_shift = jax.vmap(permutation_equivariant_fn)(bijector_feat_in)
                    log_scale_and_shift = log_scale_and_shift[jnp.arange(x.shape[0]), jnp.arange(x.shape[0])]
                else:
                    bijector_feat_in = h
                    log_scale_and_shift = permutation_equivariant_fn(bijector_feat_in)
            else:
                if condition_on_x_proj:
                    x_proj = jax.vmap(lambda x, inv_change_of_basis, origin: inv_change_of_basis @ (x - origin),
                                               in_axes=(0, 0, 0))(x, inv_change_of_basis, origin)
                    bijector_feat_in = jnp.concatenate([x_proj, h], axis=-1)
                else:
                    bijector_feat_in = h
                log_scale_and_shift = mlp_function(bijector_feat_in)

        log_scale, shift = jnp.split(log_scale_and_shift, indices_or_sections=2, axis=-1)

        chex.assert_shape(change_of_basis_matrix, (n_nodes, dim, dim))
        chex.assert_trees_all_equal_shapes(origin, log_scale, shift, x)
        return change_of_basis_matrix, origin, log_scale, shift

    def _conditioner(x, graph_features):
        chex.assert_rank(x, 3)
        n_nodes, multiplicity, dim = x.shape

        # Calculate new basis for the affine transform
        various_x_points, h = multi_x_equivariant_fn(x, graph_features)
        # various_x_points [n_nodes, multiplicity, n_vectors, dim]

        # Get info
        info = {}
        info.update(various_x_points=various_x_points)

        change_of_basis_matrix, origin, log_scale, shift = \
            jax.vmap(eq_net_out_to_params, in_axes=(1, 1, 1),
                     out_axes=1)(
            x, various_x_points, h
        )  # Vmap over multiplicity axis.
        return change_of_basis_matrix, origin, log_scale, shift, info


    def conditioner(x):
        if len(x.shape) == 3:
            return _conditioner(x, graph_features)
        else:
            assert len(x.shape) == 4
            return jax.vmap(_conditioner)(x, graph_features)

    return conditioner


def make_se_equivariant_split_coupling_with_projection(
        graph_features: chex.Array,
        layer_number: int,
        dim: int,
        n_aug: int,
        swap: bool,
        nets_config: NetsConfig,
        identity_init: bool = True,
        gram_schmidt: bool = False,
        global_frame: bool = False,
        process_flow_params_jointly: bool = False,
        condition_on_x_proj: bool = True,
        add_small_identity: bool = False
        ) -> BijectorWithExtra:
    assert n_aug % 2 == 1
    assert dim in (2, 3)  # Currently just written for 2D and 3D

    def bijector_fn(params):
        change_of_basis_matrix, origin, log_scale, shift, info = params
        return ProjectedScalarAffine(change_of_basis_matrix, origin, log_scale, shift, info)

    n_heads = dim
    n_invariant_params = dim*2

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

    if process_flow_params_jointly:
        transformer_config = nets_config.transformer_config._replace(output_dim=n_invariant_params, zero_init=identity_init)
        permutation_equivariant_fn = Transformer(name=f"layer_{layer_number}_swap{swap}_scale_shift",
                                                 config=transformer_config)
        mlp_function = None
    else:
        permutation_equivariant_fn = None
        mlp_function = hk.Sequential([
            hk.LayerNorm(axis=-1, create_offset=True, create_scale=True, param_axis=-1),
            hk.nets.MLP(nets_config.mlp_head_config.mlp_units, activate_final=True),
            hk.Linear(n_invariant_params, b_init=jnp.zeros, w_init=jnp.zeros) if identity_init else
            hk.Linear(n_invariant_params,
                      b_init=hk.initializers.VarianceScaling(0.01),
                      w_init=hk.initializers.VarianceScaling(0.01))
                                      ])

    conditioner = make_conditioner(
        graph_features=graph_features,
        global_frame=global_frame,
        process_flow_params_jointly=process_flow_params_jointly,
        mlp_function=mlp_function,
        multi_x_equivariant_fn=equivariant_fn,
        permutation_equivariant_fn=permutation_equivariant_fn,
        gram_schmidt=gram_schmidt,
        condition_on_x_proj=condition_on_x_proj,
        add_small_identity=add_small_identity
    )

    return SplitCouplingWithExtra(
        split_index=(n_aug + 1) // 2,
        event_ndims=3,  # [nodes, n_aug+1, dim]
        conditioner=conditioner,
        bijector=bijector_fn,
        swap=swap,
        split_axis=-2
    )