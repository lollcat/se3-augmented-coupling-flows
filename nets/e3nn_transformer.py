from typing import NamedTuple, Optional, Callable, Sequence
from functools import partial

import haiku as hk
import jax.numpy as jnp
import jax
from mace_jax import tools
import e3nn_jax as e3nn
from mace_jax.modules.models import LinearNodeEmbeddingBlock, RadialEmbeddingBlock, safe_norm
from e3nn_jax.experimental.transformer import Transformer as EnTransformerLayer
import chex

from utils.graph import get_senders_and_receivers_fully_connected


class EnTransformerBlockConfig(NamedTuple):
    num_heads: int
    mlp_units: Sequence[int]
    n_vectors_hidden: int
    n_invariant_feat_hidden: int
    bessel_number: int  # Number of bessel functions.
    r_max: float  # Used in bessel function.
    activation_fn: Callable = jax.nn.selu
    num_species: int = 1
    sh_irreps_max_ell: int = 3


class EnTransformerBlock(hk.Module):
    def __init__(self, config: EnTransformerBlockConfig):
        super().__init__()
        self.config = config
        # TODO: understand what is going on with the heads here.
        assert (config.n_vectors_hidden % config.num_heads) == 0
        assert (config.n_invariant_feat_hidden % config.num_heads) == 0
        self.feature_irreps = e3nn.Irreps(f"{config.n_invariant_feat_hidden}x0e")
        self.sh_irreps = e3nn.Irreps.spherical_harmonics(config.sh_irreps_max_ell)[1:]  # discard 0e
        self.sh_harms_fn = partial(
            e3nn.spherical_harmonics,
            self.sh_irreps,
            normalize=False,
            normalization="component")
        self.radial_embedding = RadialEmbeddingBlock(
            r_max=self.config.r_max,
            avg_r_min=None,
            basis_functions=lambda length, max_length: e3nn.bessel(length, x_max=max_length,
                                                                n=self.config.bessel_number),
            envelope_function=e3nn.soft_envelope,
        )

        self.transformer_fn = EnTransformerLayer(
            irreps_node_output=e3nn.Irreps(f"{self.config.n_invariant_feat_hidden}x0e+{self.config.n_vectors_hidden}x1o"),
            list_neurons=list(config.mlp_units),
            act=config.activation_fn,
            num_heads=config.num_heads
        )

    def __call__(self, positions, features):
        chex.assert_tree_shape_suffix(features, (self.config.n_invariant_feat_hidden,))
        chex.assert_tree_shape_suffix(positions, (self.config.n_vectors_hidden, 3))
        senders, receivers = get_senders_and_receivers_fully_connected(positions.shape[0])

        # Prepare the edge attributes.
        vectors = positions[senders] - positions[receivers]
        lengths = safe_norm(vectors, axis=-1)

        sph_harmon = jax.vmap(self.sh_harms_fn, in_axes=-2, out_axes=-2)(vectors / lengths[..., None])
        sph_harmon = sph_harmon.axis_to_mul()

        radial_embedding = jax.vmap(self.radial_embedding, in_axes=-2)(lengths).axis_to_mul()

        edge_attr = e3nn.concatenate([radial_embedding, sph_harmon])

        # Setup as fully connected for now.
        edge_weight_cutoff = jnp.ones(vectors.shape[:1])  # e3nn.soft_envelope(vectors, x_max=self.config.r_max)

        transformer_feat_in = e3nn.IrrepsArray(self.feature_irreps, features)
        # Pass through transformer.
        transformer_out = self.transformer_fn(edge_src=senders, edge_dst=receivers,
                                              edge_weight_cutoff=edge_weight_cutoff,
                                              edge_attr=edge_attr,
                                              node_feat=transformer_feat_in)

        # Turn back into positions and features.
        vector_features = transformer_out.filter(keep=f"{self.config.n_vectors_hidden}x1o")
        vector_features = vector_features.factor_mul_to_last_axis()  # [n_nodes, n_vectors, dim]
        vector_features = vector_features.array
        invariant_features = transformer_out.filter(keep=f"{self.config. n_invariant_feat_hidden}x0e")

        invariant_features = hk.nets.MLP((*self.config.mlp_units, self.config.n_invariant_feat_hidden),
                                         activation=self.config.activation_fn)(invariant_features.array)

        # Residual connections, then output.
        position_out = vector_features + positions
        features_out = invariant_features + features
        return position_out, features_out


class TransformerTorsoConfig(NamedTuple):
    block_config: EnTransformerBlockConfig
    n_blocks: int
    layer_stack: bool = True

class EnTransformerConfig(NamedTuple):
    name: str
    n_invariant_feat_readout: int
    n_vectors_readout: int
    zero_init_invariant_feat: bool
    torso_config: TransformerTorsoConfig


class EnTransformer(hk.Module):
    def __init__(self, config: EnTransformerConfig):
        super().__init__(name=config.name)
        self.config = config
        self.transformer_block_fn = lambda x, h: EnTransformerBlock(config=config.torso_config.block_config)(x, h)
        self.output_irreps = e3nn.Irreps(f"{config.n_invariant_feat_readout}x0e+"
                                          f"{config.n_vectors_readout}x1o")


    def __call__(self, x):
        if len(x.shape) == 2:
            return self.call_single(x)
        else:
            chex.assert_rank(x, 3)
            return hk.vmap(self.call_single, split_rng=False)(x)

    def call_single(self, x):
        n_nodes = x.shape[0]
        h = jnp.ones((n_nodes, self.config.torso_config.block_config.n_invariant_feat_hidden))
        x = jnp.stack([x]*self.config.torso_config.block_config.n_vectors_hidden, axis=-2)

        if self.config.torso_config.layer_stack:
            stack = hk.experimental.layer_stack(self.config.torso_config.n_blocks, with_per_layer_inputs=False,
                                                name="EGCL_layer_stack")
            x, h = stack(self.transformer_block_fn)(x, h)
        else:
            for i in range(self.config.torso_config.n_blocks):
                x, h = self.transformer_block_fn(x, h)

        # Get vector features for final layer.
        center_of_mass = jnp.mean(x, axis=-2, keepdims=True)
        vector_feat = e3nn.IrrepsArray('1x1o', x - center_of_mass)
        vector_feat = vector_feat.axis_to_mul(axis=-2)
        assert vector_feat.irreps == e3nn.Irreps(f"{self.config.torso_config.block_config.n_vectors_hidden}x1o")

        # Get invariant features for final layer and concatenate.
        irreps_h = e3nn.Irreps(f"{self.config.torso_config.block_config.n_invariant_feat_hidden}x0e")
        h = e3nn.IrrepsArray(irreps_h, h)
        final_layer_in = e3nn.concatenate([h, vector_feat], axis=-1)

        # Pass through final layer, and then split back into vector and invariant features.
        out = e3nn.haiku.Linear(self.output_irreps)(final_layer_in)
        vector_features = out.filter(keep=f"{self.config.n_vectors_readout}x1o")
        vector_features = vector_features.factor_mul_to_last_axis()  # [n_nodes, n_vectors, dim]
        vector_features = vector_features.array
        chex.assert_shape(vector_features, (n_nodes, self.config.n_vectors_readout, 3))
        invariant_features = out.filter(keep=f"{self.config.n_vectors_readout}x0e").array
        return vector_features + center_of_mass, invariant_features


