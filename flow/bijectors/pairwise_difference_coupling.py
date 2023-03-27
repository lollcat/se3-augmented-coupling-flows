from typing import Tuple, Callable, Union

import chex
import distrax
import jax
import jax.numpy as jnp

from utils.numerical import rotate_2d, vector_rejection, safe_norm
from flow.distrax_with_extra import BijectorWithExtra, Array, BlockWithExtra, Extra

BijectorParams = chex.Array


class SubtractSplitCoupling(BijectorWithExtra):
    def __init__(self,
               split_index: int,
               event_ndims: int,
               graph_features: chex.Array,
               conditioner: Callable[[chex.Array, chex.Array], BijectorParams],
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
        self._conditoner = conditioner
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

    def forward_and_log_det_single(self, x: Array, graph_features: chex.Array) -> Tuple[Array, Array]:
        """Computes y = f(x) and log|det J(f)(x)|."""
        self._check_forward_input_shape(x)
        x1, x2 = self._split(x)
        params = self._conditoner(x1, graph_features)
        x2_diffs = x2 - x1
        y2_diffs, logdet = self._inner_bijector(params).forward_and_log_det(x2_diffs)
        y2 = x1 + y2_diffs
        return self._recombine(x1, y2), logdet

    def inverse_and_log_det_single(self, y: Array, graph_features: chex.Array) -> Tuple[Array, Array]:
        """Computes x = f^{-1}(y) and log|det J(f^{-1})(y)|."""
        self._check_inverse_input_shape(y)
        y1, y2 = self._split(y)
        params = self._conditoner(y1, graph_features)
        y2_diffs = y2 - y1
        x2_diffs, logdet = self._inner_bijector(params).inverse_and_log_det(y2_diffs)
        x2 = y1 + x2_diffs
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


