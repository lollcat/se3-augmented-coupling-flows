from typing import Tuple, Callable, Union

import chex
import distrax
import jax
import jax.numpy as jnp

from flow.distrax_with_extra import BijectorWithExtra, Array, Extra
from flow.bijectors.along_vector_flow_layer import AlongVectorFlow

BijectorParams = chex.Array


class AlongVectorSplitCoupling(BijectorWithExtra):
    def __init__(self,
                 split_index: int,
                 graph_features: chex.Array,
                 get_reference_vectors_and_invariant_vals: Callable,
                 bijector: Callable[[BijectorParams], Union[BijectorWithExtra, distrax.Bijector]],
                 swap: bool = False,
                 use_aux_loss: bool = True,
                 n_inner_transforms: int = 1,
                 event_ndims: int = 3,
                 split_axis: int = -2,
                 ):
        if event_ndims != 3:
            raise NotImplementedError("Only implemented for 3 event ndims")
        if split_axis != -2:
            raise NotImplementedError("Only implemented for split axis on the multiplicity axis (-2")
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
        self.use_aux_loss = use_aux_loss
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

    def adjust_centering_pre_proj(self, x: chex.Array) -> chex.Array:
        """x[:, 0, :] is constrained to ZeroMean Gaussian. But if `self._swap` is True, then we
        change this constraint to be with respect to x2[:, 0, :] instead."""
        chex.assert_rank(x, 3)  # [n_nodes, multiplicity, dim]
        if self._swap:
            centre_of_mass = jnp.mean(x[:, self._split_axis], axis=0, keepdims=True)[:, None, :]
            return x - centre_of_mass
        else:
            return x

    def adjust_centering_post_proj(self, x: chex.Array) -> chex.Array:
        """Move centre of mass to be on the first multiplicity again if `self._swap` is True."""
        chex.assert_rank(x, 3)  # [n_nodes, multiplicity, dim]
        if self._swap:
            centre_of_mass = jnp.mean(x[:, 0], axis=0, keepdims=True)[:, None, :]
            return x - centre_of_mass
        else:
            return x

    def _inner_bijector(self, params: BijectorParams, reference: chex.Array,
                        vector_index: int) -> Union[BijectorWithExtra, distrax.Bijector]:
      """Returns an inner bijector for the passed params."""
      inner_bijector = self._bijector(params, vector_index)
      spherical_bijector = AlongVectorFlow(inner_bijector, reference)
      return spherical_bijector

    def get_reference_points_and_h(self, x: chex.Array, graph_features: chex.Array) ->\
            Tuple[chex.Array, chex.Array, Extra]:
        chex.assert_rank(x, 3)
        n_nodes, multiplicity, dim = x.shape
        # Calculate new basis for the affine transform
        reference_vectors, h = self._get_reference_points_and_invariant_vals(x, graph_features)
        chex.assert_shape(reference_vectors, (n_nodes, multiplicity, self.n_inner_transforms, dim))
        reference_points = x[:, :, None, :] + reference_vectors

        extra = Extra()
        return reference_points, h, extra


    def forward_and_log_det_single_with_extra(self, x: Array, graph_features: chex.Array) -> Tuple[Array, Array, Extra]:
        """Computes y = f(x) and log|det J(f)(x)|."""
        self._check_forward_input_shape(x)
        chex.assert_rank(x, 3)
        dim = x.shape[-1]

        x = self.adjust_centering_pre_proj(x)
        x1, x2 = self._split(x)
        reference_points_all, bijector_feat_in, extra = self.get_reference_points_and_h(x1, graph_features)
        n_nodes, multiplicity, n_transforms, dim_ = reference_points_all.shape
        assert n_transforms == self.n_inner_transforms

        log_det_total = jnp.zeros(())
        for i in range(self.n_inner_transforms):
            reference_points = reference_points_all[:, :, i]
            bijector = self._inner_bijector(params=bijector_feat_in, reference=reference_points, vector_index=i)
            x2, log_det = bijector.forward_and_log_det(x2)
            log_det_total = log_det_total + log_det
            chex.assert_shape(log_det_total, ())

        y2 = x2
        y = self._recombine(x1, y2)
        y = self.adjust_centering_post_proj(y)
        return y, log_det_total, extra

    def inverse_and_log_det_single_with_extra(self, y: Array, graph_features: chex.Array) -> Tuple[Array, Array, Extra]:
        """Computes x = f^{-1}(y) and log|det J(f^{-1})(y)|."""
        self._check_inverse_input_shape(y)
        dim = y.shape[-1]
        chex.assert_rank(y, 3)

        y = self.adjust_centering_pre_proj(y)
        y1, y2 = self._split(y)
        reference_points_all, bijector_feat_in, extra = self.get_reference_points_and_h(y1, graph_features)
        n_nodes, multiplicity, n_transforms, dim_ = reference_points_all.shape
        assert n_transforms == self.n_inner_transforms

        log_det_total = jnp.zeros(())
        for i in reversed(range(self.n_inner_transforms)):
            reference_points = reference_points_all[:, :, i]
            bijector = self._inner_bijector(params=bijector_feat_in, reference=reference_points, vector_index=i)
            y2, log_det = bijector.inverse_and_log_det(y2)
            log_det_total = log_det_total + log_det
            chex.assert_shape(log_det_total, ())

        x2 = y2
        x = self._recombine(y1, x2)
        x = self.adjust_centering_post_proj(x)
        return x, log_det_total, extra

    def forward_and_log_det_single(self, x: Array, graph_features: chex.Array) -> Tuple[Array, Array]:
        return self.forward_and_log_det_single_with_extra(x, graph_features)[:2]

    def inverse_and_log_det_single(self, y: Array, graph_features: chex.Array) -> Tuple[Array, Array]:
        return self.inverse_and_log_det_single_with_extra(y, graph_features)[:2]

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
            h, logdet, extra = self.forward_and_log_det_single_with_extra(x, self._graph_features)
        elif len(x.shape) == 4:
            if self._graph_features.shape[0] != x.shape[0]:
                print("graph features has no batch size")
                h, logdet, extra = jax.vmap(self.forward_and_log_det_single_with_extra, in_axes=(0, None))(
                    x, self._graph_features)
            else:
                h, logdet, extra = jax.vmap(self.forward_and_log_det_single_with_extra)(
                    x, self._graph_features)
            extra = extra._replace(aux_info=extra.aggregate_info(), aux_loss=jnp.mean(extra.aux_loss))
        else:
            raise NotImplementedError
        return h, logdet, extra

    def inverse_and_log_det_with_extra(self, y: Array) -> Tuple[Array, Array, Extra]:
        """Computes x = f^{-1}(y) and log|det J(f^{-1})(y)|."""
        if len(y.shape) == 3:
            x, logdet, extra = self.inverse_and_log_det_single_with_extra(y, self._graph_features)
        elif len(y.shape) == 4:
            if self._graph_features.shape[0] != y.shape[0]:
                print("graph features has no batch size")
                x, logdet, extra = jax.vmap(self.inverse_and_log_det_single_with_extra, in_axes=(0, None))(
                    y, self._graph_features)
            else:
                x, logdet, extra = jax.vmap(self.inverse_and_log_det_single_with_extra)(y, self._graph_features)
            extra = extra._replace(aux_info=extra.aggregate_info(), aux_loss=jnp.mean(extra.aux_loss))
        else:
            raise NotImplementedError
        return x, logdet, extra
