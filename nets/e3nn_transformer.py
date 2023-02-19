from typing import NamedTuple

import chex
import jax.numpy as jnp
import jax
import haiku as hk
import e3nn_jax as e3nn


class TransformerLayerConfig(NamedTuple):
    irreps_input: e3nn.Irreps
    irreps_query: e3nn.Irreps
    irreps_key: e3nn.Irreps
    irreps_output: e3nn.Irreps
    max_radius: float


class E3NNTransformerLayer(hk.Module):
    """Largely following https://docs.e3nn.org/en/stable/guide/transformer.html."""
    def __init__(self, config: TransformerLayerConfig):
        super().__init__()
        self.config = config


    def __call__(self,
        features: jnp.ndarray,
        positions: jnp.ndarray):

        pass

