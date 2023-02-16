from typing import Tuple, Optional, Callable, Sequence

import distrax
import chex
import jax
import jax.numpy as jnp
import haiku as hk

from nets.nets import EgnnConfig, Transformer, TransformerConfig
from nets.nets_multi_x import MultiEgnnConfig, multi_se_equivariant_net
from flow.bijectors.bijector_proj_real_nvp import get_new_space_basis


def perform_low_rank_matmul(points, scale, vectors):
    """Assumes points are 1D with shape (number of nodes * dimensionality of euclidian space,)."""
    chex.assert_rank(points, 1)
    chex.assert_rank(vectors, 2)
    chex.assert_equal_shape((points, scale, vectors[:, 0]))
    result = points * scale + (points @ vectors) @ vectors.T  # (num_particles*3, )
    chex.assert_equal_shape((result, points))
    return result


def perform_low_rank_matmul_inverse(points, scale, vectors):
    chex.assert_rank(points, 1)
    chex.assert_rank(vectors, 2)
    chex.assert_equal_shape((points, scale, vectors[:, 0]))

    n_vectors = vectors.shape[-1]

    scaled_point = points / scale
    matrix_to_invert = (vectors.T * scale[None, :]**(-1)) @ vectors + jnp.eye(n_vectors)
    solve = jax.scipy.linalg.solve(matrix_to_invert, scaled_point @ vectors, assume_a='pos')
    woodbury_correction = (solve @ vectors.T) / scale

    transformed_points = scaled_point - woodbury_correction
    return transformed_points

def project(points, origin, change_of_basis):
    return change_of_basis.T @ (points - origin)

def un_project(points, origin, change_of_basis):
    return change_of_basis @ points + origin


def reshape_things_for_low_rank_matmul(points, scale):
    """Flatten along nodes and dimension in order to prepare for low rank matmul."""
    chex.assert_rank(points, 2)
    chex.assert_equal_shape((points, scale))
    points = points.flatten()
    scale = scale.flatten()
    return points, scale


def matmul_in_invariant_space(points, change_of_basis_matrix, origin, scale, shift, vectors):
    """Perform affine transformation in the space define by the `origin` and `change_of_basis_matrix`, and then
    go back into the original space."""
    chex.assert_rank(points, 2)
    N, D = points.shape
    chex.assert_rank(vectors, 2)
    chex.assert_rank(change_of_basis_matrix, 2)
    chex.assert_equal_shape((points, scale, shift))
    chex.assert_equal_shape((points[0], origin))

    point_in_new_space = jax.vmap(project, in_axes=(0, None, None))(points, origin, change_of_basis_matrix)
    # Reshape.
    point_in_new_space, scale = reshape_things_for_low_rank_matmul(point_in_new_space, scale)
    transformed_point_in_new_space = perform_low_rank_matmul(point_in_new_space, scale, vectors)
    # Unflatten and apply shift
    transformed_point_in_new_space = transformed_point_in_new_space.reshape((N, D)) + shift
    new_point_original_space = jax.vmap(un_project, in_axes=(0, None, None))(transformed_point_in_new_space,
                                                                             origin, change_of_basis_matrix)
    return new_point_original_space


def inverse_matmul_in_invariant_space(points, change_of_basis_matrix, origin, scale, shift, vectors):
    """Inverse of `matmul_in_invariant_space`."""
    chex.assert_rank(points, 2)
    N, D = points.shape
    chex.assert_rank(vectors, 2)
    chex.assert_rank(change_of_basis_matrix, 2)
    chex.assert_equal_shape((points, scale, shift))
    chex.assert_equal_shape((points[0], origin))

    point_in_new_space = jax.vmap(project, in_axes=(0, None, None))(points, origin, change_of_basis_matrix)
    # Inverse shift, and flatten things.
    points_before_low_rank_matmul_inv, scale = reshape_things_for_low_rank_matmul(
        point_in_new_space - shift, scale)
    transformed_point_in_new_space = perform_low_rank_matmul_inverse(points_before_low_rank_matmul_inv, scale, vectors)

    # Unflatten and project back into original space.
    transformed_point_in_new_space = transformed_point_in_new_space.reshape((N, D))
    new_point_original_space = jax.vmap(un_project, in_axes=(0, None, None))(transformed_point_in_new_space, origin,
                                                                          change_of_basis_matrix)
    return new_point_original_space

class ProjectedScalarAffine(distrax.Bijector):
    """Following style of `ScalarAffine` distrax Bijector.

    Note: Doesn't need to operate on batches, as it gets called with vmap."""
    def __init__(self, change_of_basis_matrix, origin, log_scale, shift, vectors, activation=jax.nn.softplus):
        super().__init__(event_ndims_in=1, is_constant_jacobian=True)
        self._change_of_basis_matrix = change_of_basis_matrix
        self._origin = origin
        self._shift = shift
        self._vectors = vectors
        self._n_vectors = vectors.shape[-1]

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
            self._log_scale = jnp.log(self._scale)



    @property
    def shift(self) -> chex.Array:
        return self._shift

    @property
    def log_scale(self) -> chex.Array:
        return self._log_scale

    @property
    def scale(self) -> chex.Array:
        return self._scale

    @property
    def change_of_basis_matrix(self) -> chex.Array:
        return self._change_of_basis_matrix

    @property
    def origin(self) -> chex.Array:
        return self._origin

    def forward(self, x: chex.Array) -> chex.Array:
        """Computes y = f(x)."""
        if len(x.shape) == 2:
            return matmul_in_invariant_space(x, self._change_of_basis_matrix, self._origin, self._scale,
                                                       self._shift, self._vectors)
        elif len(x.shape) == 3:
            return jax.vmap(matmul_in_invariant_space)(x, self._change_of_basis_matrix, self._origin, self._scale,
                                                                 self._shift, self._vectors)
        else:
            raise Exception

    def forward_log_det_jacobian_single(self, vectors, scale, log_scale) -> chex.Array:
        """Computes log|det J(f)(x)|."""
        n_vectors = vectors.shape[1]
        scale = scale.flatten()
        slog_det_in = jnp.eye(n_vectors) + vectors.T @ (vectors * (scale[:, None] ** -1))
        s, log_det2 = jnp.linalg.slogdet(slog_det_in)
        return jnp.sum(log_scale, axis=-1) + log_det2

    def forward_log_det_jacobian(self, x: chex.Array) -> chex.Array:
        if len(x.shape) == 2:
            log_det = self.forward_log_det_jacobian_single(self._vectors, self._scale, self._log_scale)
        else:
            log_det = jax.vmap(self.forward_log_det_jacobian_single)(self._vectors, self._scale, self._log_scale)
        return log_det

    def forward_and_log_det(self, x: chex.Array) -> Tuple[chex.Array, chex.Array]:
        """Computes y = f(x) and log|det J(f)(x)|."""
        return self.forward(x), self.forward_log_det_jacobian(x)

    def inverse(self, y: chex.Array) -> chex.Array:
        """Computes x = f^{-1}(y)."""
        if len(y.shape) == 2:
            return inverse_matmul_in_invariant_space(y, self._change_of_basis_matrix, self._origin, self._scale,
                                                               self._shift, self._vectors)
        elif len(y.shape) == 3:
            return jax.vmap(inverse_matmul_in_invariant_space)(
                y, self._change_of_basis_matrix, self._origin, self._scale, self._shift, self._vectors)
        else:
            raise Exception

    def inverse_log_det_jacobian(self, y: chex.Array) -> chex.Array:
        """Computes log|det J(f^{-1})(y)|."""
        return - self.forward_log_det_jacobian(y)

    def inverse_and_log_det(self, y: chex.Array) -> Tuple[chex.Array, chex.Array]:
        """Computes x = f^{-1}(y) and log|det J(f^{-1})(y)|."""
        return self.inverse(y), self.inverse_log_det_jacobian(y)


def make_conditioner(
        n_vectors: int,
        process_flow_params_jointly: bool,
        multi_x_equivariant_fn: Callable,
        permutation_equivariant_fn: Optional[Callable] = None,
        mlp_function: Optional[Callable] = None,
        gram_schmidt: bool = False,
        condition_on_x_proj: bool = False,
                     ):
    if process_flow_params_jointly:
        assert permutation_equivariant_fn is not None
    else:
        assert mlp_function is not None

    def _conditioner(x):
        chex.assert_rank(x, 2)
        n_nodes, dim = x.shape

        # Calculate new basis for the affine transform
        various_x_points, h = multi_x_equivariant_fn(x)

        origin, change_of_basis_matrix = get_new_space_basis(x, various_x_points, gram_schmidt, global_frame=True)

        inv_change_of_basis = change_of_basis_matrix.T  # jnp.linalg.inv(change_of_basis_matrix)
        if condition_on_x_proj:
            x_proj = jax.vmap(lambda x, inv_change_of_basis, origin: inv_change_of_basis @ (x - origin),
                              in_axes=(0, None, None))(x, inv_change_of_basis, origin)
            bijector_feat_in = jnp.concatenate([x_proj, h], axis=-1)
        else:
            bijector_feat_in = h
        if process_flow_params_jointly:
            invariant_bijector_params = permutation_equivariant_fn(bijector_feat_in)
        else:
            invariant_bijector_params = mlp_function(bijector_feat_in)

        log_scale = invariant_bijector_params[..., :dim]
        shift = invariant_bijector_params[..., dim:dim*2]
        vectors_params = invariant_bijector_params[..., dim*2:]
        chex.assert_tree_shape_suffix(vectors_params, (dim*n_vectors,))
        vectors = jnp.reshape(vectors_params, (n_nodes*dim, n_vectors))

        chex.assert_shape(change_of_basis_matrix, (dim, dim))
        chex.assert_trees_all_equal_shapes(log_scale, shift, x)
        chex.assert_trees_all_equal_shapes(origin, x[0])
        return change_of_basis_matrix, origin, log_scale, shift, vectors

    def conditioner(x):
        if len(x.shape) == 2:
            return _conditioner(x)
        else:
            assert len(x.shape) == 3
            return jax.vmap(_conditioner)(x)

    return conditioner


def make_se_equivariant_split_coupling_with_projection(layer_number, dim, swap,
                                                       egnn_config: EgnnConfig,
                                                       transformer_config: Optional[TransformerConfig] = None,
                                                       identity_init: bool = True,
                                                       gram_schmidt: bool = False,
                                                       process_flow_params_jointly: bool = True,
                                                       condition_on_x_proj: bool = False,
                                                       mlp_function_units: Optional[Sequence[int]] = None,
                                                       n_vectors: Optional[int] = None,
                                                       ):
    assert dim in (2, 3)  # Currently just written for 2D and 3D
    n_vectors = n_vectors if n_vectors else 10  # should not hardcode this and make it non-optional
    n_invariant_params = dim*2 + dim*n_vectors

    def bijector_fn(params):
        change_of_basis_matrix, origin, log_scale, shift, vectors = params
        return ProjectedScalarAffine(change_of_basis_matrix, origin, log_scale, shift, vectors)

    n_heads = dim + (1 if gram_schmidt else 0)
    egnn_config = egnn_config._replace(name=f"layer_{layer_number}_swap{swap}_multi_x_egnn",
                                       identity_init_x=False, zero_init_h=identity_init,
                                       h_config=egnn_config.h_config._replace(h_out=True,
                                       h_out_dim=egnn_config.h_config.h_embedding_dim))
    multi_egnn_config = MultiEgnnConfig(egnn_config=egnn_config, n_heads=n_heads)
    multi_egnn = multi_se_equivariant_net(multi_egnn_config)


    if process_flow_params_jointly:
        transformer_config = transformer_config._replace(output_dim=n_invariant_params, zero_init=identity_init)
        permutation_equivariant_fn = Transformer(name=f"layer_{layer_number}_swap{swap}_scale_shift",
                                                 config=transformer_config)
        mlp_function = None
    else:
        permutation_equivariant_fn = None
        mlp_function = hk.Sequential([
            hk.LayerNorm(axis=-1, create_offset=True, create_scale=True, param_axis=-1),
            hk.nets.MLP(mlp_function_units, activate_final=True),
            hk.Linear(n_invariant_params, b_init=jnp.zeros, w_init=jnp.zeros) if identity_init else
            hk.Linear(n_invariant_params,
                      b_init=hk.initializers.VarianceScaling(0.01),
                      w_init=hk.initializers.VarianceScaling(0.01))
                                      ])

    conditioner = make_conditioner(
        n_vectors=n_vectors,
        process_flow_params_jointly=process_flow_params_jointly,
        mlp_function=mlp_function,
        multi_x_equivariant_fn=multi_egnn,
        permutation_equivariant_fn=permutation_equivariant_fn,
        gram_schmidt=gram_schmidt,
        condition_on_x_proj=condition_on_x_proj
    )

    return distrax.SplitCoupling(
        split_index=dim,
        event_ndims=2,  # [nodes, dim]
        conditioner=conditioner,
        bijector=bijector_fn,
        swap=swap,
        split_axis=-1
    )
