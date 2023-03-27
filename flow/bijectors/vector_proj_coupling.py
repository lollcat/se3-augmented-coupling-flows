from typing import Tuple, Callable, Union

import chex
import distrax
import jax
import jax.numpy as jnp
import tensorflow_probability.substrates.jax as tfp

from utils.numerical import safe_norm
from flow.distrax_with_extra import BijectorWithExtra, Array, BlockWithExtra, Extra

BijectorParams = chex.Array

class VectorProjSplitCoupling(BijectorWithExtra):
    def __init__(self,
                 split_index: int,
                 event_ndims: int,
                 graph_features: chex.Array,
                 get_reference_vectors_and_invariant_vals: Callable,
                 bijector: Callable[[BijectorParams], Union[BijectorWithExtra, distrax.Bijector]],
                 swap: bool = False,
                 split_axis: int = -1,
                 add_small_identity: bool = True,
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
        self._split_index = split_index
        self._bijector = bijector
        self._swap = swap
        self._split_axis = split_axis
        self._get_reference_points_and_invariant_vals = get_reference_vectors_and_invariant_vals
        self._graph_features = graph_features
        self._add_small_identity = add_small_identity
        super().__init__(event_ndims_in=event_ndims)


    @property
    def norm_to_unconstrained_bijector(self):
        return distrax.Block(distrax.Inverse(tfp.bijectors.Exp()), 3)

    def _split(self, x: Array) -> Tuple[Array, Array]:
        x1, x2 = jnp.split(x, [self._split_index], self._split_axis)
        if self._swap:
          x1, x2 = x2, x1
        return x1, x2

    def _recombine(self, x1: Array, x2: Array) -> Array:
        if self._swap:
          x1, x2 = x2, x1
        return jnp.concatenate([x1, x2], self._split_axis)

    def _inner_bijector(self, params: BijectorParams, vector_index: int) -> Union[BijectorWithExtra, distrax.Bijector]:
      """Returns an inner bijector for the passed params."""
      bijector = self._bijector(params, vector_index)
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

    def get_reference_points_and_h(self, x: chex.Array, graph_features: chex.Array) ->\
            Tuple[chex.Array, chex.Array, Extra]:
        chex.assert_rank(x, 3)
        n_nodes, multiplicity, dim = x.shape

        # Calculate new basis for the affine transform
        reference_vectors, h = self._get_reference_points_and_invariant_vals(x, graph_features)
        reference_points = x[:, :, None, :] + reference_vectors
        # TODO: Can calculate distrances to reference points and input this to bijectors.
        bijector_feat_in = h
        extra = Extra()  # self.get_extra(vectors)
        return reference_points, bijector_feat_in, extra

    def forward_and_log_det_single(self, x: Array, graph_features: chex.Array) -> Tuple[Array, Array]:
        """Computes y = f(x) and log|det J(f)(x)|."""
        self._check_forward_input_shape(x)
        dim = x.shape[-1]
        x1, x2 = self._split(x)
        reference_points, bijector_feat_in, _ = self.get_reference_points_and_h(x1, graph_features)
        n_nodes, multiplicity, n_vectors, dim_ = reference_points.shape
        log_det_total = jnp.zeros(())
        for i in range(n_vectors):
            reference_point = reference_points[:, :, i, :]
            chex.assert_equal_shape((x1, reference_point))
            vector = x2 - reference_point
            norms_in = safe_norm(vector, axis=-1, keepdims=True)
            normed_vector = vector / norms_in
            unconstrained_in, log_det_unconstrained_fwd = self.norm_to_unconstrained_bijector.forward_and_log_det(
                norms_in)
            unconstrained_out, logdet_inner_bijector = \
                self._inner_bijector(bijector_feat_in, i).forward_and_log_det(unconstrained_in)
            chex.assert_equal_shape((unconstrained_out, unconstrained_in))
            norms_out, log_det_unconstrained_rv = self.norm_to_unconstrained_bijector.inverse_and_log_det(unconstrained_out)
            x2 = reference_point + norms_out * normed_vector
            log_det_total = log_det_total + logdet_inner_bijector + log_det_unconstrained_fwd + log_det_unconstrained_rv
            chex.assert_shape(log_det_total, ())
        y2 = x2
        return self._recombine(x1, y2), log_det_total*dim

    def inverse_and_log_det_single(self, y: Array, graph_features: chex.Array) -> Tuple[Array, Array]:
        """Computes x = f^{-1}(y) and log|det J(f^{-1})(y)|."""
        self._check_inverse_input_shape(y)
        dim = y.shape[-1]
        y1, y2 = self._split(y)
        reference_points, bijector_feat_in, _ = self.get_reference_points_and_h(y1, graph_features)
        n_nodes, multiplicity, n_vectors, dim_ = reference_points.shape
        log_det_total = jnp.zeros(())
        for i in reversed(range(n_vectors)):
            reference_point = reference_points[:, :, i, :]
            chex.assert_equal_shape((y2, reference_point))
            vector = y2 - reference_point
            norms_in = safe_norm(vector, axis=-1, keepdims=True)
            normed_vector = vector / norms_in
            unconstrained_in, log_det_unconstrained_fwd = self.norm_to_unconstrained_bijector.forward_and_log_det(norms_in)
            log_det_total = log_det_total
            unconstrained_out, logdet_inner_bijector = \
                self._inner_bijector(bijector_feat_in, i).inverse_and_log_det(unconstrained_in)
            norms_out, log_det_unconstrained_rv = self.norm_to_unconstrained_bijector.inverse_and_log_det(unconstrained_out)
            y2 = reference_point + norms_out * normed_vector
            log_det_total = log_det_total + logdet_inner_bijector + log_det_unconstrained_fwd + log_det_unconstrained_rv
            chex.assert_shape(log_det_total, ())
        x2 = y2
        return self._recombine(y1, x2), log_det_total*dim

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

    # def forward_and_log_det_with_extra_single(self, x: Array, graph_features: chex.Array) -> Tuple[Array, Array, Extra]:
    #     """Computes y = f(x) and log|det J(f)(x)|."""
    #     self._check_forward_input_shape(x)
    #     x1, x2 = self._split(x)
    #     origin, change_of_basis_matrix, bijector_feat_in, extra = self.get_reference_points_and_h(x1, graph_features)
    #     x2_proj = jax.vmap(jax.vmap(project))(x2, origin, change_of_basis_matrix)
    #     y2, logdet = self._inner_bijector(bijector_feat_in).forward_and_log_det(x2_proj)
    #     y2 = jax.vmap(jax.vmap(unproject))(y2, origin, change_of_basis_matrix)
    #     return self._recombine(x1, y2), logdet, extra
    #
    # def inverse_and_log_det_with_extra_single(self, y: Array, graph_features: chex.Array) -> Tuple[Array, Array, Extra]:
    #     """Computes x = f^{-1}(y) and log|det J(f^{-1})(y)|."""
    #     self._check_inverse_input_shape(y)
    #     y1, y2 = self._split(y)
    #     origin, change_of_basis_matrix, bijector_feat_in, extra = self.get_reference_points_and_h(y1, graph_features)
    #     y2_proj = jax.vmap(jax.vmap(project))(y2, origin, change_of_basis_matrix)
    #     x2, logdet = self._inner_bijector(bijector_feat_in).inverse_and_log_det(y2_proj)
    #     x2 = jax.vmap(jax.vmap(unproject))(x2, origin, change_of_basis_matrix)
    #     return self._recombine(y1, x2), logdet, extra
    #
    # def forward_and_log_det_with_extra(self, x: Array) -> Tuple[Array, Array, Extra]:
    #     """Computes y = f(x) and log|det J(f)(x)|."""
    #     if len(x.shape) == 3:
    #         h, logdet, extra = self.forward_and_log_det_with_extra_single(x, self._graph_features)
    #     elif len(x.shape) == 4:
    #         if self._graph_features.shape[0] != x.shape[0]:
    #             print("graph features has no batch size")
    #             h, logdet, extra = jax.vmap(self.forward_and_log_det_with_extra_single, in_axes=(0, None))(
    #                 x, self._graph_features)
    #         else:
    #             h, logdet, extra = jax.vmap(self.forward_and_log_det_with_extra_single)(
    #                 x, self._graph_features)
    #         extra = extra._replace(aux_info=extra.aggregate_info(), aux_loss=jnp.mean(extra.aux_loss))
    #     else:
    #         raise NotImplementedError
    #     return h, logdet, extra
    #
    # def inverse_and_log_det_with_extra(self, y: Array) -> Tuple[Array, Array, Extra]:
    #     """Computes x = f^{-1}(y) and log|det J(f^{-1})(y)|."""
    #     if len(y.shape) == 3:
    #         x, logdet, extra = self.inverse_and_log_det_with_extra_single(y, self._graph_features)
    #     elif len(y.shape) == 4:
    #         if self._graph_features.shape[0] != y.shape[0]:
    #             print("graph features has no batch size")
    #             x, logdet, extra = jax.vmap(self.inverse_and_log_det_with_extra_single, in_axes=(0, None))(
    #                 y, self._graph_features)
    #         else:
    #             x, logdet, extra = jax.vmap(self.inverse_and_log_det_with_extra_single)(y, self._graph_features)
    #         extra = extra._replace(aux_info=extra.aggregate_info(), aux_loss=jnp.mean(extra.aux_loss))
    #     else:
    #         raise NotImplementedError
    #     return x, logdet, extra


#
#
# def get_min_k_vectors_by_norm(norms, vectors, receivers, n_vectors, node_index):
#     _, min_k_indices = jax.lax.top_k(-norms[receivers == node_index], n_vectors)
#     min_k_vectors = vectors[receivers == node_index][min_k_indices]
#     return min_k_vectors
#
#
# def get_directions_for_closest_atoms(x: chex.Array, n_vectors: int) -> chex.Array:
#     chex.assert_rank(x, 2)  # [n_nodes, dim]
#     n_nodes, dim = x.shape
#     senders, receivers = get_senders_and_receivers_fully_connected(dim)
#     vectors = x[receivers] - x[senders]
#     norms = safe_norm(vectors, axis=-1, keepdims=False)
#     min_k_vectors = jax.vmap(get_min_k_vectors_by_norm, in_axes=(None, None, None, None, 0))(
#         norms, vectors, receivers, n_vectors, jnp.arange(n_nodes))
#     chex.assert_shape(min_k_vectors, (n_nodes, n_vectors, dim))
#     return min_k_vectors
