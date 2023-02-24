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



class EnTransformerBlock(hk.Module):
    def __init__(self,
                num_heads: int,
                mlp_units: Sequence[int],
                n_vectors_hidden: int,
                n_invariant_feat_hidden: int,
                bessel_number: int,  # Number of bessel functions.
                r_max: float,  # Used in bessel function.
                raw_distance_in_radial_embedding: bool = False,
                node_feat_as_edge_feat: bool = False,
                activation_fn: Callable = jax.nn.silu,
                num_species: int = 1,
                sh_irreps_max_ell: int = 3):
        super().__init__()
        self.num_heads = num_heads
        self.mlp_units = mlp_units
        self.n_vectors_hidden = n_vectors_hidden
        self.n_invariant_feat_hidden = n_invariant_feat_hidden
        self.bessel_number = bessel_number
        self.r_max = r_max
        # Allows node featuers to go into edge MLP, similar to egnn.
        self.node_feat_as_edge_feat = node_feat_as_edge_feat
        # Whether to also pass in the length as 0e edge info.
        self.raw_distance_in_radial_embedding = raw_distance_in_radial_embedding
        self.activation_fn = activation_fn
        self.num_species = num_species
        self.sh_irreps_max_ell = sh_irreps_max_ell
        # TODO: understand what is going on with the heads here.
        assert (n_vectors_hidden % num_heads) == 0
        assert (n_invariant_feat_hidden % num_heads) == 0
        self.feature_irreps = e3nn.Irreps(f"{n_invariant_feat_hidden}x0e")
        self.sh_irreps = e3nn.Irreps.spherical_harmonics(sh_irreps_max_ell)[1:]  # discard 0e
        self.sh_harms_fn = partial(
            e3nn.spherical_harmonics,
            self.sh_irreps,
            normalize=False,
            normalization="component")
        self.radial_embedding = RadialEmbeddingBlock(
            r_max=self.r_max,
            avg_r_min=None,
            basis_functions=lambda length, max_length: e3nn.bessel(length, x_max=max_length,
                                                                n=self.bessel_number),
            envelope_function=e3nn.soft_envelope,
        )

        self.transformer_fn = EnTransformerLayer(
            irreps_node_output=e3nn.Irreps(f"{self.n_invariant_feat_hidden}x0e+{self.n_vectors_hidden}x1o"),
            list_neurons=list(mlp_units),
            act=activation_fn,
            num_heads=num_heads
        )

    def __call__(self, positions, features):
        chex.assert_tree_shape_suffix(features, (self.n_invariant_feat_hidden,))
        chex.assert_tree_shape_suffix(positions, (self.n_vectors_hidden, 3))
        senders, receivers = get_senders_and_receivers_fully_connected(positions.shape[0])

        # Prepare the edge attributes.
        vectors = positions[senders] - positions[receivers]
        lengths = safe_norm(vectors, axis=-1)

        sph_harmon = jax.vmap(self.sh_harms_fn, in_axes=-2, out_axes=-2)(vectors / lengths[..., None])
        sph_harmon = sph_harmon.axis_to_mul()

        radial_embedding = jax.vmap(self.radial_embedding, in_axes=-2)(lengths).axis_to_mul()
        if self.raw_distance_in_radial_embedding:
            raw_dist_info = jax.nn.softmax(lengths, axis=-2)
            scalar_edge_features = e3nn.concatenate([radial_embedding, e3nn.IrrepsArray(f"{self.n_vectors_hidden}x0e",
                                                                                    raw_dist_info)]).simplify()
        else:
            scalar_edge_features = radial_embedding
        if self.node_feat_as_edge_feat:
            scalar_edge_features = e3nn.concatenate([
                scalar_edge_features,
                e3nn.IrrepsArray(f"{self.n_invariant_feat_hidden}x0e", features[senders]),
                e3nn.IrrepsArray(f"{self.n_invariant_feat_hidden}x0e", features[receivers]),
            ]).simplify()
        scalar_edge_features = e3nn.haiku.MultiLayerPerceptron(list_neurons=list(self.mlp_units),
                                                           act=self.activation_fn,
                                                           output_activation=self.activation_fn)(scalar_edge_features)

        edge_attr = e3nn.concatenate([scalar_edge_features, sph_harmon]).simplify()

        # Setup as fully connected for now.
        edge_weight_cutoff = jnp.ones(vectors.shape[:1])  # e3nn.soft_envelope(vectors, x_max=self.r_max)

        transformer_feat_in = e3nn.IrrepsArray(self.feature_irreps, features)
        # Pass through transformer.
        transformer_out = self.transformer_fn(edge_src=senders, edge_dst=receivers,
                                              edge_weight_cutoff=edge_weight_cutoff,
                                              edge_attr=edge_attr,
                                              node_feat=transformer_feat_in)

        # Turn back into positions and features.
        vector_features = transformer_out.filter(keep=f"{self.n_vectors_hidden}x1o")
        vector_features = vector_features.factor_mul_to_last_axis()  # [n_nodes, n_vectors, dim]
        vector_features = vector_features.array
        invariant_features = transformer_out.filter(keep=f"{self. n_invariant_feat_hidden}x0e")

        invariant_features = hk.nets.MLP((*self.mlp_units, self.n_invariant_feat_hidden),
                                         activation=self.activation_fn)(invariant_features.array)

        # Residual connections, then output.
        position_out = vector_features + positions
        features_out = invariant_features + features
        return position_out, features_out


class EnTransformerTorsoConfig(NamedTuple):
    n_blocks: int
    mlp_units: Sequence[int]
    n_vectors_hidden: int
    n_invariant_feat_hidden: int
    bessel_number: int  # Number of bessel functions.
    r_max: float  # Used in bessel function.
    raw_distance_in_radial_embedding: bool = False
    node_feat_as_edge_feat: bool = False
    num_heads: int = 1
    activation_fn: Callable = jax.nn.silu
    num_species: int = 1
    sh_irreps_max_ell: int = 3
    layer_stack: bool = True

    def get_transformer_layer_kwargs(self):
        kwargs = {}
        kwargs.update(num_heads=self.num_heads,
                        mlp_units=self.mlp_units,
                        n_vectors_hidden=self.n_vectors_hidden,
                        n_invariant_feat_hidden=self.n_invariant_feat_hidden,
                        bessel_number = self.bessel_number,
                        r_max = self.r_max,
                        activation_fn = self.activation_fn,
                        num_species = self.num_species,
                        sh_irreps_max_ell= self.sh_irreps_max_ell,
                        raw_distance_in_radial_embedding=self.raw_distance_in_radial_embedding,
                        node_feat_as_edge_feat=self.node_feat_as_edge_feat
        )
        return kwargs

class EnTransformerConfig(NamedTuple):
    name: str
    n_invariant_feat_readout: int
    n_vectors_readout: int
    zero_init_invariant_feat: bool
    torso_config: EnTransformerTorsoConfig


class EnTransformer(hk.Module):
    def __init__(self, config: EnTransformerConfig):
        super().__init__(name=config.name)
        self.config = config
        self.transformer_block_fn = lambda x, h: EnTransformerBlock(
            **config.torso_config.get_transformer_layer_kwargs())(x, h)
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
        h = jnp.ones((n_nodes, self.config.torso_config.n_invariant_feat_hidden))
        x = jnp.stack([x]*self.config.torso_config.n_vectors_hidden, axis=-2)

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
        assert vector_feat.irreps == e3nn.Irreps(f"{self.config.torso_config.n_vectors_hidden}x1o")

        # Get invariant features for final layer and concatenate.
        irreps_h = e3nn.Irreps(f"{self.config.torso_config.n_invariant_feat_hidden}x0e")
        h = e3nn.IrrepsArray(irreps_h, h)
        final_layer_in = e3nn.concatenate([h, vector_feat], axis=-1)

        # Pass through final layer.
        out = e3nn.haiku.Linear(self.output_irreps)(final_layer_in)

        # Get vector features.
        vector_features = out.filter(keep=f"{self.config.n_vectors_readout}x1o")
        vector_features = vector_features.factor_mul_to_last_axis()  # [n_nodes, n_vectors, dim]
        vector_features = vector_features.array
        chex.assert_shape(vector_features, (n_nodes, self.config.n_vectors_readout, 3))

        # Get scalar features.
        invariant_features = out.filter(keep=f"{self.config.n_vectors_readout}x0e")
        invariant_features = hk.Linear(invariant_features.shape[-1],
                                       w_init=jnp.zeros if self.config.zero_init_invariant_feat else None,
                                       )(jax.nn.elu(invariant_features.array))
        return vector_features + center_of_mass, invariant_features


