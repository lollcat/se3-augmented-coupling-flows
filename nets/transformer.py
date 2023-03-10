from typing import NamedTuple, Optional, Sequence

import haiku as hk
import chex
import jax
import jax.numpy as jnp

class TransformerConfig(NamedTuple):
    output_dim: Optional[int] = None
    num_heads: int = 3
    key_size: int = 4
    w_init_scale: float = 0.1
    mlp_units: Sequence[int] = (32, 32)
    n_layers: int = 3
    zero_init: bool = False
    layer_norm: bool = True


class TransformerBlock(hk.Module):
    # Largely follows: https://theaisummer.com/jax-transformer/
    def __init__(self, name: str, config: TransformerConfig = TransformerConfig()):
        super().__init__(name=name)
        self.config = config

    def __call__(self, x):
        # Simplifying assumption for now to make residual connections and layer stacking easy.
        chex.assert_tree_shape_suffix(x, (self.config.key_size*self.config.num_heads,))

        x_in = hk.LayerNorm(axis=-1, create_scale=True, create_offset=True)(x) if self.config.layer_norm else x
        x_attn = hk.MultiHeadAttention(num_heads=self.config.num_heads, key_size=self.config.key_size,
                                       w_init=hk.initializers.VarianceScaling(self.config.w_init_scale))(
            x_in, x_in, x_in)
        x = x + x_attn
        x_in = hk.LayerNorm(axis=-1, create_scale=True, create_offset=True)(x) if self.config.layer_norm else x
        x_dense = hk.nets.MLP([*self.config.mlp_units, self.config.num_heads*self.config.key_size])(x_in)
        x = x + x_dense
        return x


class Transformer(hk.Module):
    def __init__(self, name: str, config: TransformerConfig = TransformerConfig()):
        super().__init__(name=name + "_vanilla_transformer")
        self.config = config
        self.transformer_blocks = [TransformerBlock(name=name + 'vanilla_transformer' + str(i),
                                                    config=config) for i in range(self.config.n_layers)]

    def __call__(self, x):
        x_out = jax.nn.relu(hk.Linear(self.config.num_heads * self.config.key_size)(x))
        for transformer in self.transformer_blocks:
                x_out = transformer(x_out)
        if self.config.output_dim is not None:
            final_layer = hk.Linear(self.config.output_dim, w_init=jnp.zeros, b_init=jnp.zeros) \
                if self.config.zero_init else hk.Linear(self.config.output_dim,
                                                        w_init=hk.initializers.VarianceScaling(0.01),
                                                        b_init=hk.initializers.VarianceScaling(0.01))
            x_out = final_layer(x_out)
        return x_out
