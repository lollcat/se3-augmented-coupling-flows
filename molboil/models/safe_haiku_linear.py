from typing import Optional, Union, Sequence, Any

import haiku as hk
import jax
import jax.numpy as jnp
import jax.lax as lax
import numpy as np


class TruncatedNormal(hk.initializers.Initializer):
  # Prevent TPU error. with jax.random.truncated_normal
  """Initializes by sampling from a truncated normal distribution."""

  def __init__(self,
               stddev: Union[float, jnp.ndarray] = 1.,
               mean: Union[float, jnp.ndarray] = 0.):
    """Constructs a :class:`TruncatedNormal` initializer.

    Args:
      stddev: The standard deviation parameter of the truncated
        normal distribution.
      mean: The mean of the truncated normal distribution.
    """
    self.stddev = stddev
    self.mean = mean

  def __call__(self, shape: Sequence[int], dtype: Any) -> jnp.ndarray:
    real_dtype = jnp.finfo(dtype).dtype
    m = jax.lax.convert_element_type(self.mean, dtype)
    s = jax.lax.convert_element_type(self.stddev, real_dtype)
    is_complex = jnp.issubdtype(dtype, jnp.complexfloating)
    if is_complex:
      shape = [2, *shape]
    unscaled = jax.random.truncated_normal(hk.next_rng_key(), -2., 2., shape,
                                           jnp.float32).astype(real_dtype)
    if is_complex:
      unscaled = unscaled[0] + 1j * unscaled[1]
    return s * unscaled + m

class Linear(hk.Module):
  """Linear module."""

  def __init__(
      self,
      output_size: int,
      with_bias: bool = True,
      w_init: Optional[hk.initializers.Initializer] = None,
      b_init: Optional[hk.initializers.Initializer] = None,
      name: Optional[str] = None,
  ):
    """Constructs the Linear module.

    Args:
      output_size: Output dimensionality.
      with_bias: Whether to add a bias to the output.
      w_init: Optional initializer for weights. By default, uses random values
        from truncated normal, with stddev ``1 / sqrt(fan_in)``. See
        https://arxiv.org/abs/1502.03167v3.
      b_init: Optional initializer for bias. By default, zero.
      name: Name of the module.
    """
    super().__init__(name=name)
    self.input_size = None
    self.output_size = output_size
    self.with_bias = with_bias
    self.w_init = w_init
    self.b_init = b_init or jnp.zeros

  def __call__(
      self,
      inputs: jnp.ndarray,
      *,
      precision: Optional[lax.Precision] = None,
  ) -> jnp.ndarray:
    """Computes a linear transform of the input."""
    if not inputs.shape:
      raise ValueError("Input must not be scalar.")

    input_size = self.input_size = inputs.shape[-1]
    output_size = self.output_size
    dtype = inputs.dtype

    w_init = self.w_init
    if w_init is None:
      stddev = 1. / np.sqrt(self.input_size)
      w_init = TruncatedNormal(stddev=stddev)
    w = hk.get_parameter("w", [input_size, output_size], dtype, init=w_init)

    out = jnp.dot(inputs, w, precision=precision)

    if self.with_bias:
      b = hk.get_parameter("b", [self.output_size], dtype, init=self.b_init)
      b = jnp.broadcast_to(b, out.shape)
      out = out + b

    return out