from typing import Tuple, Callable, Union

import chex
import distrax
import jax
import jax.numpy as jnp

from flow.distrax_with_extra import BijectorWithExtra, Array, Extra
from flow.aug_flow_dist import GraphFeatures, Positions

BijectorParams = chex.Array

class CentreOfMassInvariantSplitCoupling(BijectorWithExtra):
    """Only permutation and centre of mass equivariant. Not rotation equivariance flow layer."""
    def __init__(self,
                 split_index: int,
                 graph_features: chex.Array,
                 conditioner: Callable[[Positions, GraphFeatures], BijectorParams],
                 bijector: Callable[[BijectorParams, int], Union[BijectorWithExtra, distrax.Bijector]],
                 swap: bool = False,
                 n_inner_transforms: int = 1,
                 event_ndims: int = 3,
                 split_axis: int = -2,
                 ):
        super().__init__(event_ndims_in=event_ndims, is_constant_jacobian=False)
        if event_ndims != 3:
            raise NotImplementedError("Only implemented for 3 event ndims")
        if split_axis != -2:
            raise NotImplementedError("Only implemented for split axis on the multiplicity axis (-2")
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
        self._graph_features = graph_features
        self._conditioner = conditioner
        self.n_inner_transforms = n_inner_transforms
        super().__init__(event_ndims_in=event_ndims)

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

    def _split(self, x: chex.Array) -> Tuple[chex.Array, chex.Array]:
        x1, x2 = jnp.split(x, [self._split_index], self._split_axis)
        if self._swap:
            x1, x2 = x2, x1
        return x1, x2

    def _recombine(self, x1: chex.Array, x2: chex.Array) -> chex.Array:
        if self._swap:
          x1, x2 = x2, x1
        return jnp.concatenate([x1, x2], self._split_axis)

    def _inner_bijector(self, params: BijectorParams, transform_index: int) -> \
            Union[BijectorWithExtra, distrax.Bijector]:
      """Returns an inner bijector for the passed params."""
      inner_inner_bijector = self._bijector(params, transform_index)  # e.g. spline or rnvp
      assert inner_inner_bijector.event_ndims_in == inner_inner_bijector.event_ndims_out == 0
      return distrax.Block(inner_inner_bijector, ndims=3)

    def forward_and_log_det_with_extra_single(self, x: Array, graph_features: chex.Array) -> Tuple[Array, Array, Extra]:
        """Computes y = f(x) and log|det J(f)(x)|."""
        chex.assert_rank(x, 3)
        dim = x.shape[-1]

        self._check_forward_input_shape(x)
        x = self.adjust_centering_pre_proj(x)
        x1, x2 = self._split(x)
        bijector_feat_in = self._conditioner(x1, graph_features)
        chex.assert_rank(bijector_feat_in, 2)
        n_nodes, multiplicity_dim = bijector_feat_in.shape

        log_det_total = jnp.zeros(())
        for i in range(self.n_inner_transforms):
            inner_bijector = self._inner_bijector(bijector_feat_in, i)
            x2, log_det_inner = inner_bijector.forward_and_log_det(x2)
            log_det_total = log_det_total + log_det_inner

        y2 = x2
        y = self._recombine(x1, y2)
        y = self.adjust_centering_post_proj(y)
        return y, log_det_total, Extra()

    def inverse_and_log_det_with_extra_single(self, y: Array, graph_features: chex.Array) -> Tuple[Array, Array, Extra]:
        """Computes x = f^{-1}(y) and log|det J(f^{-1})(y)|."""
        self._check_inverse_input_shape(y)
        chex.assert_rank(y, 3)
        dim = y.shape[-1]

        y = self.adjust_centering_pre_proj(y)
        y1, y2 = self._split(y)
        bijector_feat_in = self._conditioner(y1, graph_features)
        n_nodes, multiplicity_dim = bijector_feat_in.shape

        log_det_total = jnp.zeros(())
        for i in reversed(range(self.n_inner_transforms)):
            inner_bijector = self._inner_bijector(bijector_feat_in, i)
            y2, log_det_inner = inner_bijector.inverse_and_log_det(y2)
            log_det_total = log_det_total + log_det_inner

        x2 = y2
        x = self._recombine(y1, x2)
        x = self.adjust_centering_post_proj(x)
        return x, log_det_total, Extra()

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