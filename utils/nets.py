"""From /acme/jax/networks/continuous.py"""

from typing import Sequence

import haiku as hk
import jax
import jax.numpy as jnp

uniform_initializer = hk.initializers.UniformScaling(scale=0.333)

class NearZeroInitializedLinear(hk.Linear):
  """Simple linear layer, initialized at near zero weights and zero biases."""

  def __init__(self, output_size: int, scale: float = 1e-4):
    super().__init__(output_size, w_init=hk.initializers.VarianceScaling(scale))


class LayerNormMLP(hk.Module):
  """Simple feedforward MLP torso with initial layer-norm.
  This module is an MLP which uses LayerNorm (with a tanh normalizer) on the
  first layer and non-linearities (elu) on all but the last remaining layers.
  """

  def __init__(self, layer_sizes: Sequence[int], activate_final: bool = False):
    """Construct the MLP.
    Args:
      layer_sizes: a sequence of ints specifying the size of each layer.
      activate_final: whether or not to use the activation function on the final
        layer of the neural network.
    """
    super().__init__(name='feedforward_mlp_torso')

    self._network = hk.Sequential([
        hk.Linear(layer_sizes[0], w_init=uniform_initializer),
        hk.LayerNorm(axis=-1, create_scale=True, create_offset=True),
        jax.lax.tanh,
        hk.nets.MLP(
            layer_sizes[1:],
            w_init=uniform_initializer,
            activation=jax.nn.elu,
            activate_final=activate_final),
    ])

  def __call__(self, inputs: jnp.ndarray) -> jnp.ndarray:
    """Forwards the policy network."""
    return self._network(inputs)