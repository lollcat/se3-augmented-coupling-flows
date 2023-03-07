"""Copy distrax Bijector chain but use hk scan to make it fast to compile.
Also allows for extra info from within the bijector to be passed forwards.
"""

from typing import Tuple, Callable

import chex
from distrax._src.bijectors.bijector import Array
import haiku as hk
import jax.numpy as jnp

from flow.distrax_with_info import BijectorWithInfo, Extra

class Chain(BijectorWithInfo):
  """Chain of the same bijector, that is fast to compile. Also allows for extra info being returned."""

  def __init__(self, bijector_fn: Callable[[], BijectorWithInfo], n_layers, compile_n_unroll=1):
    self._bijector_fn = bijector_fn
    self._n_layers = n_layers
    self.stack = hk.experimental.layer_stack(self._n_layers, with_per_layer_inputs=False, name="flow_layer_stack",
                                            unroll=compile_n_unroll)
    self.stack_with_info = hk.experimental.layer_stack(self._n_layers, with_per_layer_inputs=True,
                                                       name="flow_layer_stack", unroll=compile_n_unroll)

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

  def single_forward_fn_with_extra(self, carry, _):
    x, log_det = carry
    y, log_det_new, extra = self._bijector_fn().forward_and_log_det_with_extra(x)
    chex.assert_equal_shape((x, y))
    chex.assert_equal_shape((log_det_new, log_det))
    return (y, log_det + log_det_new), extra

  def single_reverse_fn_with_extra(self, carry, _):
    y, log_det = carry
    x, log_det_new, extra = self._bijector_fn().inverse_and_log_det_with_extra(y)
    chex.assert_equal_shape((y, x))
    chex.assert_equal_shape((log_det_new, log_det))
    return (x, log_det + log_det_new), extra

  def forward_and_log_det_with_extra(self, x: Array) -> Tuple[Array, Array, Extra]:
    """Computes y = f(x) and log|det J(f)(x)|."""
    log_det_init = jnp.zeros(x.shape[0:-self.event_ndims_in])
    (x_out, log_det), extra = self.stack_with_info(self.single_forward_fn_with_extra)((x, log_det_init),
                                                                                      jnp.zeros(self._n_layers),
                                                                                      reverse=True)
    return x_out, log_det, extra

  def inverse_and_log_det_with_extra(self, y: Array) -> Tuple[Array, Array, Extra]:
    """Computes x = f^{-1}(y) and log|det J(f^{-1})(y)|."""
    log_det_init = jnp.zeros(y.shape[0:-self.event_ndims_in])
    (x_out, log_det), extra = self.stack_with_info(self.single_reverse_fn_with_extra,
                                                   )((y, log_det_init, jnp.zeros(self._n_layers)))
    return x_out, log_det, extra
