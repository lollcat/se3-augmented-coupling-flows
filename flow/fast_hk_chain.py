"""Copy distrax Bijector chain but use hk scan to make it fast to compile."""

from typing import List, Tuple, Callable

import chex
from distrax._src.bijectors import bijector as base
import haiku as hk
import jax.numpy as jnp

Array = base.Array
BijectorLike = base.BijectorLike
BijectorT = base.BijectorT


class Chain(base.Bijector):
  """Chain of the same bijector, that is fast to compile."""

  def __init__(self, bijector_fn: Callable, n_layers, compile_n_unroll=2):
    self._bijector_fn = bijector_fn
    self._n_layers = n_layers
    self.stack = hk.experimental.layer_stack(self._n_layers, with_per_layer_inputs=False, name="flow_layer_stack",
                                            unroll=compile_n_unroll)

    is_constant_jacobian = False
    is_constant_log_det = False
    super().__init__(
        event_ndims_in=2,
        event_ndims_out=2,
        is_constant_jacobian=is_constant_jacobian,
        is_constant_log_det=is_constant_log_det)

  def single_forward_fn(self, x, log_det):
    y, log_det_new = self._bijector_fn().forward_and_log_det(x)
    chex.assert_equal_shape((x, y))
    chex.assert_equal_shape((log_det_new, log_det))
    return y, log_det + log_det_new

  def single_reverse_fn(self, y, log_det):
    x, log_det_new = self._bijector_fn().inverse_and_log_det(y)
    chex.assert_equal_shape((y, x))
    chex.assert_equal_shape((log_det_new, log_det))
    return x, log_det + log_det_new


  def forward_and_log_det(self, x: Array) -> Tuple[Array, Array]:
    """Computes y = f(x) and log|det J(f)(x)|."""
    log_det_init = jnp.zeros(x.shape[0:-self.event_ndims_in])
    x_out, log_det = self.stack(self.single_forward_fn)(x, log_det_init, reverse=True)
    return x_out, log_det

  def inverse_and_log_det(self, y: Array) -> Tuple[Array, Array]:
    """Computes x = f^{-1}(y) and log|det J(f^{-1})(y)|."""
    log_det_init = jnp.zeros(y.shape[0:-self.event_ndims_in])
    x_out, log_det = self.stack(self.single_reverse_fn)(y, log_det_init)
    return x_out, log_det
