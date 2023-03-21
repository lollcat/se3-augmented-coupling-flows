from typing import Tuple, Callable, Union

import chex
import distrax
import jax
import jax.numpy as jnp

from utils.numerical import rotate_2d, vector_rejection, safe_norm
from flow.distrax_with_extra import BijectorWithExtra, Array, BlockWithExtra

BijectorParams = chex.Array


def project(x, origin, change_of_basis_matrix):
    chex.assert_rank(x, 1)
    chex.assert_rank(change_of_basis_matrix, 2)
    chex.assert_equal_shape((x, origin, change_of_basis_matrix[0], change_of_basis_matrix[:, 0]))
    return change_of_basis_matrix.T @ (x - origin)

def unproject(x, origin, change_of_basis_matrix):
    chex.assert_rank(x, 1)
    chex.assert_rank(change_of_basis_matrix, 2)
    chex.assert_equal_shape((x, origin, change_of_basis_matrix[0], change_of_basis_matrix[:, 0]))
    return change_of_basis_matrix @ x + origin


def get_new_space_basis(x: chex.Array, various_x_vectors: chex.Array, add_small_identity: bool = False):
    n_nodes, dim = x.shape
    # Calculate new basis for the affine transform
    various_x_vectors = jnp.swapaxes(various_x_vectors, 0, 1)

    origin = x + various_x_vectors[0]
    basis_vectors = various_x_vectors[1:]

    if add_small_identity:
        # Add independant vectors to try help improve numerical stability
        basis_vectors = basis_vectors + jnp.eye(x.shape[-1])[:basis_vectors.shape[0]][:, None, :]*1e-6
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
    chex.assert_equal_shape((origin, x))
    chex.assert_shape(change_of_basis_matrix, (n_nodes, dim, dim))
    return origin, change_of_basis_matrix


class ProjSplitCoupling(BijectorWithExtra):
  def __init__(self,
               split_index: int,
               event_ndims: int,
               graph_features: chex.Array,
               get_basis_vectors_and_invariant_vals: Callable,
               bijector: Callable[[BijectorParams], Union[BijectorWithExtra, distrax.Bijector]],
               swap: bool = False,
               split_axis: int = -1):
    super().__init__(event_ndims_in=event_ndims, is_constant_jacobian=False)
    if split_index < 0:
      raise ValueError(
          f'The split index must be non-negative; got {split_index}.')
    if split_axis >= 0:
      raise ValueError(f'The split axis must be negative; got {split_axis}.')
    if event_ndims < 0:
      raise ValueError(
          f'`event_ndims` must be non-negative; got {event_ndims}.')
    if split_axis < -event_ndims:
      raise ValueError(
          f'The split axis points to an axis outside the event. With '
          f'`event_ndims == {event_ndims}`, the split axis must be between -1 '
          f'and {-event_ndims}. Got `split_axis == {split_axis}`.')
    self._split_index = split_index
    self._bijector = bijector
    self._swap = swap
    self._split_axis = split_axis
    self._get_basis_vectors_and_invariant_vals = get_basis_vectors_and_invariant_vals
    self._graph_features = graph_features
    super().__init__(event_ndims_in=event_ndims)

  def _split(self, x: Array) -> Tuple[Array, Array]:
    x1, x2 = jnp.split(x, [self._split_index], self._split_axis)
    if self._swap:
      x1, x2 = x2, x1
    return x1, x2

  def _recombine(self, x1: Array, x2: Array) -> Array:
    if self._swap:
      x1, x2 = x2, x1
    return jnp.concatenate([x1, x2], self._split_axis)

  def _inner_bijector(self, params: BijectorParams) -> Union[BijectorWithExtra, distrax.Bijector]:
      """Returns an inner bijector for the passed params."""
      bijector = self._bijector(params)
      if bijector.event_ndims_in != bijector.event_ndims_out:
          raise ValueError(
              f'The inner bijector must have `event_ndims_in==event_ndims_out`. '
              f'Instead, it has `event_ndims_in=={bijector.event_ndims_in}` and '
              f'`event_ndims_out=={bijector.event_ndims_out}`.')
      extra_ndims = self.event_ndims_in - bijector.event_ndims_in
      if extra_ndims < 0:
          raise ValueError(
              f'The inner bijector can\'t have more event dimensions than the '
              f'coupling bijector. Got {bijector.event_ndims_in} for the inner '
              f'bijector and {self.event_ndims_in} for the coupling bijector.')
      elif extra_ndims > 0:
          if isinstance(bijector, BijectorWithExtra):
              bijector = BlockWithExtra(bijector, extra_ndims)
          else:
              bijector = distrax.Block(bijector, extra_ndims)
      return bijector

  def get_basis_and_h(self, x, graph_features: chex.Array):
      chex.assert_rank(x, 3)
      n_nodes, multiplicity, dim = x.shape

      # Calculate new basis for the affine transform
      various_x_points, h = self._get_basis_vectors_and_invariant_vals(x, graph_features)
      # Vmap over multiplicity.
      origin, change_of_basis_matrix = jax.vmap(get_new_space_basis, in_axes=1, out_axes=1)(x, various_x_points)

      # Stack h, and x projected into the space.
      x_proj = jax.vmap(jax.vmap(project))(x, origin, change_of_basis_matrix)
      bijector_feat_in = jnp.concatenate([x_proj, h], axis=-1)
      return origin, change_of_basis_matrix, bijector_feat_in

  def forward_and_log_det_single(self, x: Array, graph_features: chex.Array) -> Tuple[Array, Array]:
    """Computes y = f(x) and log|det J(f)(x)|."""
    self._check_forward_input_shape(x)
    x1, x2 = self._split(x)
    origin, change_of_basis_matrix, bijector_feat_in = self.get_basis_and_h(x1, graph_features)
    x2_proj = jax.vmap(jax.vmap(project))(x2, origin, change_of_basis_matrix)
    y2, logdet = self._inner_bijector(bijector_feat_in).forward_and_log_det(x2_proj)
    y2 = jax.vmap(jax.vmap(unproject))(y2, origin, change_of_basis_matrix)
    return self._recombine(x1, y2), logdet

  def inverse_and_log_det_single(self, y: Array, graph_features: chex.Array) -> Tuple[Array, Array]:
    """Computes x = f^{-1}(y) and log|det J(f^{-1})(y)|."""
    self._check_inverse_input_shape(y)
    y1, y2 = self._split(y)
    origin, change_of_basis_matrix, bijector_feat_in = self.get_basis_and_h(y1, graph_features)
    y2_proj = jax.vmap(jax.vmap(project))(y2, origin, change_of_basis_matrix)
    x2, logdet = self._inner_bijector(bijector_feat_in).inverse_and_log_det(y2_proj)
    x2 = jax.vmap(jax.vmap(unproject))(x2, origin, change_of_basis_matrix)
    return self._recombine(y1, x2), logdet

  def forward_and_log_det(self, x: Array) -> Tuple[Array, Array]:
    """Computes y = f(x) and log|det J(f)(x)|."""
    if len(x.shape) == 3:
        return self.forward_and_log_det_single(x, self._graph_features)
    elif len(x.shape) == 4:
        if self._graph_features.shape[0] != x.shape[0]:
            print("graph features has no batch size")
            return jax.vmap(self.forward_and_log_det_single, in_axes=(0, None))(x, self._graph_features)
        else:
            return jax.vmap(self.forward_and_log_det_single)(x, self._graph_features)
    else:
        raise NotImplementedError

  def inverse_and_log_det(self, y: Array) -> Tuple[Array, Array]:
    """Computes x = f^{-1}(y) and log|det J(f^{-1})(y)|."""
    if len(y.shape) == 3:
        return self.inverse_and_log_det_single(y, self._graph_features)
    elif len(y.shape) == 4:
        if self._graph_features.shape[0] != y.shape[0]:
            print("graph features has no batch size")
            return jax.vmap(self.inverse_and_log_det_single, in_axes=(0, None))(y, self._graph_features)
        else:
            return jax.vmap(self.inverse_and_log_det_single)(y, self._graph_features)
    else:
        raise NotImplementedError