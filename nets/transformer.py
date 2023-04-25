from typing import NamedTuple, Sequence

import haiku as hk
import chex
import jax
import jax.numpy as jnp

class TransformerBlockConfig(NamedTuple):
    name: str
    key_size: int
    mlp_units: Sequence[int]
    n_layers: int
    num_heads: int = 3
    w_init_scale: float = 0.1
    zero_init: bool = False
    layer_norm: bool = True


class TransformerBlock(hk.Module):
    """Largely follows: https://theaisummer.com/jax-transformer/"""
    def __init__(self, config: TransformerBlockConfig):
        super().__init__(name=config.name)
        self.config = config

    def __call__(self, x: chex.Array) -> chex.Array:
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


class TransformerConfig(NamedTuple):
    output_dim: int
    key_size_per_node_dim_in: int  # key_size = multiplicity * dim * key_size_per_node_dim_in
    n_layers: int
    mlp_units: Sequence[int]
    num_heads: int = 3
    w_init_scale: float = 0.1
    zero_init: bool = False
    layer_norm: bool = True

    def get_block_config(self, node_dimension: int, layer_number: int):
        input_dict = self._asdict()
        del(input_dict["output_dim"])
        key_size_per_node_dim_in =input_dict.pop('key_size_per_node_dim_in')
        key_size = key_size_per_node_dim_in*node_dimension
        name = f"layer_{layer_number}"
        return TransformerBlockConfig(name=name, key_size=key_size, **input_dict)


class Transformer(hk.Module):
    """A transformer that takes in positions and features."""
    def __init__(self, name: str, config: TransformerConfig):
        super().__init__(name=name + "_vanilla_transformer")
        self.config = config

    def __call__(self, positions: chex.Array, features: chex.Array) -> chex.Array:
        chex.assert_rank(positions, 3)
        chex.assert_rank(features, 3)
        chex.assert_axis_dimension(features, 1, 1)
        n_nodes, multiplicity, dim = positions.shape

        # Flatten multiplicity and dim dimensions into each other.
        positions = jnp.reshape(positions, (positions.shape[0], multiplicity*dim))
        features = jnp.squeeze(features, axis=1)
        x = jnp.concatenate([positions, features], axis=-1)
        # Use linear layer to project into space with same width as transformer.
        layer_width = self.config.num_heads * self.config.key_size_per_node_dim_in*multiplicity*dim
        x_out = jax.nn.relu(hk.Linear(layer_width)(x))

        # Pass through self-attention layers.
        for i in range(self.config.n_layers):
            transformer_block = TransformerBlock(self.config.get_block_config(node_dimension=multiplicity*dim,
                                                                              layer_number=i))
            x_out = transformer_block(x_out)

        # Final linear layer.
        final_layer = hk.Linear(self.config.output_dim, w_init=jnp.zeros, b_init=jnp.zeros) \
            if self.config.zero_init else hk.Linear(self.config.output_dim,
                                                    w_init=hk.initializers.VarianceScaling(0.01),
                                                    b_init=hk.initializers.VarianceScaling(0.01))
        x_out = final_layer(x_out)
        chex.assert_rank(x_out, 2)
        chex.assert_axis_dimension(x_out, 0, n_nodes)

        return x_out
