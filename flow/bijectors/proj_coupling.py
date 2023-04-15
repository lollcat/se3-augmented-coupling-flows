from typing import Tuple, Callable, Union

import chex
import distrax
import jax
import jax.numpy as jnp

from molboil.utils.numerical import rotate_2d

from utils.numerical import vector_rejection, safe_norm
from flow.distrax_with_extra import BijectorWithExtra, Array, Extra
from flow.bijectors.zero_mean_bijector import ZeroMeanBijector
from flow.bijectors.proj_flow_layer import ProjFlow

BijectorParams = chex.Array




def get_equivariant_orthonormal_basis(vectors: chex.Array, add_small_identity: bool = True,
                                      method: str = 'gram-schmidt') -> chex.Array:
    """Takes in a set of (non-orthonormal vectors), and returns an orthonormal basis, with equivariant
    vectors as it's columns."""

    assert method in ('gram-schmidt', 'loewdin')

    n_nodes, n_vectors, dim = vectors.shape

    if add_small_identity:
        # Add independant vectors to try help improve numerical stability
        vectors = vectors + jnp.eye(n_vectors, dim)[None, :, :] * 1e-6

    if method == 'gram-schmidt':
        # Set n_vectors to leading axis to make slicing simpler.
        basis_vectors = jnp.swapaxes(vectors, 0, 1)

        z_basis_vector = basis_vectors[0]
        z_basis_vector = z_basis_vector / safe_norm(z_basis_vector, axis=-1, keepdims=True)
        if dim == 3:
            # Vector rejection to get second axis orthogonal to z axis.
            x_basis_vector = basis_vectors[1]
            x_basis_vector = x_basis_vector / safe_norm(x_basis_vector, axis=-1, keepdims=True)
            x_basis_vector = vector_rejection(x_basis_vector, z_basis_vector)
            x_basis_vector = x_basis_vector / safe_norm(x_basis_vector, axis=-1, keepdims=True)

            # Cross product of z and x vector to get final vector.
            y_basis_vector = jnp.cross(z_basis_vector, x_basis_vector)
            y_basis_vector = y_basis_vector / safe_norm(y_basis_vector, axis=-1, keepdims=True)
            change_of_basis_matrix = jnp.stack([z_basis_vector, x_basis_vector, y_basis_vector], axis=-1)
        else:
            assert dim == 2
            y_basis_vector = rotate_2d(z_basis_vector, theta=jnp.pi * 0.5)
            y_basis_vector = y_basis_vector / safe_norm(y_basis_vector, axis=-1, keepdims=True)
            change_of_basis_matrix = jnp.stack([z_basis_vector, y_basis_vector], axis=-1)
    elif method == 'loewdin':
        if n_vectors == dim - 1:
            c = jnp.cross(vectors[:, :1, :], vectors[:, 1:, :])
            vectors = jnp.concatenate((vectors, c), axis=1)
        w, s, vt = jnp.linalg.svd(vectors, full_matrices=False)
        change_of_basis_matrix = jax.vmap(jnp.matmul)(w, vt)

    chex.assert_shape(change_of_basis_matrix, (n_nodes, dim, dim))
    return change_of_basis_matrix


class ProjSplitCoupling(BijectorWithExtra):
    def __init__(self,
                 split_index: int,
                 event_ndims: int,
                 graph_features: chex.Array,
                 get_basis_vectors_and_invariant_vals: Callable,
                 bijector: Callable[[BijectorParams, int], Union[BijectorWithExtra, distrax.Bijector]],
                 origin_on_coupled_pair: bool = True,
                 swap: bool = False,
                 split_axis: int = -1,
                 add_small_identity: bool = True,
                 orthogonalization_method: str = 'loewdin',
                 n_inner_transforms: int = 1,
                 ):
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
        self._origin_on_aug = origin_on_coupled_pair
        self._split_index = split_index
        self._bijector = bijector
        self._swap = swap
        self._split_axis = split_axis
        self._get_basis_vectors_and_invariant_vals = get_basis_vectors_and_invariant_vals
        self._graph_features = graph_features
        self._add_small_identity = add_small_identity
        self._orthogonalization_method = orthogonalization_method
        self.n_inner_transforms = n_inner_transforms
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

    def _inner_bijector(self, params: BijectorParams, transform_index: int,
                        origin: chex.Array, change_of_basis_matrix: chex.Array) -> \
            Union[BijectorWithExtra, distrax.Bijector]:
      """Returns an inner bijector for the passed params."""
      inner_inner_bijector = self._bijector(params, transform_index)  # e.g. spline or rnvp
      bijector = ProjFlow(inner_bijector=inner_inner_bijector, origin=origin,
                          change_of_basis_matrix=change_of_basis_matrix)
      bijector = ZeroMeanBijector(bijector)
      return bijector

    def get_basis_and_h(self, x: chex.Array, graph_features: chex.Array) ->\
            Tuple[chex.Array, chex.Array, chex.Array, Extra]:
        chex.assert_rank(x, 3)
        n_nodes, multiplicity, dim = x.shape

        # Calculate new basis for the affine transform
        vectors_out, h = self._get_basis_vectors_and_invariant_vals(x, graph_features)
        chex.assert_rank(vectors_out, 5)
        chex.assert_tree_shape_prefix(vectors_out, (n_nodes, multiplicity, self.n_inner_transforms))

        chex.assert_rank(h, 2)
        if self._origin_on_aug:
            origin = jnp.repeat(x[:, :, None], self.n_inner_transforms, axis=2)
            vectors = vectors_out
        else:
            origin = x[:, :, None] + vectors_out[:, :, :, 0]
            vectors = vectors_out[:, :, :, 1:]

        # Vmap over multiplicity.
        change_of_basis_matrices = jax.vmap(jax.vmap(get_equivariant_orthonormal_basis, in_axes=(1, None, None), out_axes=1),
        in_axes=(1, None, None), out_axes=1)(vectors, self._add_small_identity, self._orthogonalization_method)
        extra = self.get_extra(vectors)

        chex.assert_shape(origin, (n_nodes, multiplicity, self.n_inner_transforms, dim))
        chex.assert_shape(change_of_basis_matrices, (n_nodes, multiplicity, self.n_inner_transforms, dim, dim))
        return origin, change_of_basis_matrices, h, extra

    def get_vector_info_single(self, basis_vectors: chex.Array) -> Tuple[chex.Array, chex.Array, chex.Array]:
        chex.assert_rank(basis_vectors, 2)
        n_vectors, dim = basis_vectors.shape
        assert dim == 3
        assert n_vectors == 2 or n_vectors == 3
        basis_vectors = basis_vectors + jnp.eye(dim)[:n_vectors] * 1e-30
        vec1 = basis_vectors[0]
        vec2 = basis_vectors[1]
        arccos_in = jnp.dot(vec1, vec2) / safe_norm(vec1, axis=-1) / safe_norm(vec2, axis=-1)
        theta = jnp.arccos(arccos_in)
        log_barrier_in = 1 - jnp.abs(arccos_in)
        log_barrier_in = jnp.where(log_barrier_in < 1e-6, log_barrier_in + 1e-6, log_barrier_in)
        aux_loss = - jnp.log(log_barrier_in)
        return theta, aux_loss, log_barrier_in

    def get_extra(self, various_x_points: chex.Array) -> Extra:
        dim = various_x_points.shape[-1]
        if dim == 2:
            return Extra()
        else:
            # Vmap over n_nodes, multiplicity, and n_inner_transforms.
            theta, aux_loss, log_barrier_in = jax.vmap(jax.vmap(jax.vmap(self.get_vector_info_single)))(various_x_points)
            info = {}
            info_aggregator = {}
            info_aggregator.update(
                mean_abs_theta=jnp.mean, min_abs_theta=jnp.min,
                min_log_barrier_in=jnp.min
            )
            info.update(
                mean_abs_theta=jnp.mean(jnp.abs(theta)), min_abs_theta=jnp.min(jnp.abs(theta)),
                min_log_barrier_in=jnp.min(log_barrier_in)
            )
            aux_loss = jnp.mean(aux_loss)
            extra = Extra(aux_loss=aux_loss, aux_info=info, info_aggregator=info_aggregator)
            return extra

    def forward_and_log_det_with_extra_single(self, x: Array, graph_features: chex.Array) -> Tuple[Array, Array, Extra]:
        """Computes y = f(x) and log|det J(f)(x)|."""
        chex.assert_rank(x, 3)
        dim = x.shape[-1]
        x = x - jnp.mean(x, axis=-3)  # Zero mean so that reference makes sense.

        self._check_forward_input_shape(x)
        x1, x2 = self._split(x)
        origins, change_of_basis_matrices, bijector_feat_in, extra = self.get_basis_and_h(x1, graph_features)
        n_nodes, multiplicity, n_transforms, n_vectors, dim = change_of_basis_matrices.shape
        assert n_transforms == self.n_inner_transforms

        log_det_total = jnp.zeros(())
        for i in range(self.n_inner_transforms):
            origin = origins[:, :, i]
            change_of_basis_matrix = change_of_basis_matrices[:, :, i]
            inner_bijector = self._inner_bijector(bijector_feat_in, i, origin, change_of_basis_matrix)
            x2, log_det_inner = inner_bijector.forward_and_log_det(x2)
            log_det_total = log_det_total + log_det_inner

        y2 = x2
        return self._recombine(x1, y2), log_det_total, extra

    def inverse_and_log_det_with_extra_single(self, y: Array, graph_features: chex.Array) -> Tuple[Array, Array, Extra]:
        """Computes x = f^{-1}(y) and log|det J(f^{-1})(y)|."""
        self._check_inverse_input_shape(y)
        chex.assert_rank(y, 3)
        dim = y.shape[-1]
        y = y - jnp.mean(y, axis=-3)  # Zero mean so that reference makes sense.

        y1, y2 = self._split(y)
        origins, change_of_basis_matrices, bijector_feat_in, extra = self.get_basis_and_h(y1, graph_features)
        n_nodes, multiplicity, n_transforms, n_vectors, dim = change_of_basis_matrices.shape
        assert n_transforms == self.n_inner_transforms

        log_det_total = jnp.zeros(())
        for i in reversed(range(self.n_inner_transforms)):
            origin = origins[:, :, i]
            change_of_basis_matrix = change_of_basis_matrices[:, :, i]
            inner_bijector = self._inner_bijector(bijector_feat_in, i, origin, change_of_basis_matrix)
            y2, log_det_inner = inner_bijector.inverse_and_log_det(y2)
            log_det_total = log_det_total + log_det_inner

        x2 = y2
        return self._recombine(y1, x2), log_det_total, extra

    def forward_and_log_det_single(self, x: Array, graph_features: chex.Array) -> Tuple[Array, Array]:
        return self.forward_and_log_det_with_extra_single(x, graph_features)[:2]

    def inverse_and_log_det_single(self, y: Array, graph_features: chex.Array) -> Tuple[Array, Array]:
        return self.inverse_and_log_det_with_extra_single(y, graph_features)[:2]

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


    def forward_and_log_det_with_extra(self, x: Array) -> Tuple[Array, Array, Extra]:
        """Computes y = f(x) and log|det J(f)(x)|."""
        if len(x.shape) == 3:
            h, logdet, extra = self.forward_and_log_det_with_extra_single(x, self._graph_features)
        elif len(x.shape) == 4:
            if self._graph_features.shape[0] != x.shape[0]:
                print("graph features has no batch size")
                h, logdet, extra = jax.vmap(self.forward_and_log_det_with_extra_single, in_axes=(0, None))(
                    x, self._graph_features)
            else:
                h, logdet, extra = jax.vmap(self.forward_and_log_det_with_extra_single)(
                    x, self._graph_features)
            extra = extra._replace(aux_info=extra.aggregate_info(), aux_loss=jnp.mean(extra.aux_loss))
        else:
            raise NotImplementedError
        return h, logdet, extra

    def inverse_and_log_det_with_extra(self, y: Array) -> Tuple[Array, Array, Extra]:
        """Computes x = f^{-1}(y) and log|det J(f^{-1})(y)|."""
        if len(y.shape) == 3:
            x, logdet, extra = self.inverse_and_log_det_with_extra_single(y, self._graph_features)
        elif len(y.shape) == 4:
            if self._graph_features.shape[0] != y.shape[0]:
                print("graph features has no batch size")
                x, logdet, extra = jax.vmap(self.inverse_and_log_det_with_extra_single, in_axes=(0, None))(
                    y, self._graph_features)
            else:
                x, logdet, extra = jax.vmap(self.inverse_and_log_det_with_extra_single)(y, self._graph_features)
            extra = extra._replace(aux_info=extra.aggregate_info(), aux_loss=jnp.mean(extra.aux_loss))
        else:
            raise NotImplementedError
        return x, logdet, extra