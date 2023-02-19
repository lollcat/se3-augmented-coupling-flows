import functools
import math
from typing import Callable, Optional, Union

import haiku as hk
import jax
import jax.numpy as jnp
import e3nn_jax as e3nn
from mace_jax.modules.models import MACELayer, LinearNodeEmbeddingBlock, RadialEmbeddingBlock, safe_norm
try:
    from profile_nn_jax import profile
except ImportError:

    def profile(_, x):
        return x


class MACE(hk.Module):
    """The classic MACE block, but without stacking across layers."""
    def __init__(
        self,
        *,
        output_irreps: e3nn.Irreps,  # Irreps of the output, default 1x0e
        r_max: float,
        num_interactions: int,  # Number of interactions (layers), default 2
        hidden_irreps: e3nn.Irreps,  # 256x0e or 128x0e + 128x1o
        readout_mlp_irreps: e3nn.Irreps,  # Hidden irreps of the MLP in last readout, default 16x0e
        avg_num_neighbors: float,
        num_species: int,
        num_features: int = None,  # Number of features per node, default gcd of hidden_irreps multiplicities
        avg_r_min: float = None,
        radial_basis: Callable[[jnp.ndarray], jnp.ndarray],
        radial_envelope: Callable[[jnp.ndarray], jnp.ndarray],
        # Number of zero derivatives at small and large distances, default 4 and 2
        # If both are None, it uses a smooth C^inf envelope function
        max_ell: int = 3,  # Max spherical harmonic degree, default 3
        epsilon: Optional[float] = None,
        correlation: int = 3,  # Correlation order at each layer (~ node_features^correlation), default 3
        gate: Callable = jax.nn.silu,  # activation function
        symmetric_tensor_product_basis: bool = True,
        off_diagonal: bool = False,
        interaction_irreps: Union[str, e3nn.Irreps] = "o3_restricted",  # or o3_full
        node_embedding: hk.Module = LinearNodeEmbeddingBlock,
    ):
        super().__init__()

        output_irreps = e3nn.Irreps(output_irreps)
        hidden_irreps = e3nn.Irreps(hidden_irreps)
        readout_mlp_irreps = e3nn.Irreps(readout_mlp_irreps)

        if num_features is None:
            self.num_features = functools.reduce(
                math.gcd, (mul for mul, _ in hidden_irreps)
            )
            self.hidden_irreps = e3nn.Irreps(
                [(mul // self.num_features, ir) for mul, ir in hidden_irreps]
            )
        else:
            self.num_features = num_features
            self.hidden_irreps = hidden_irreps

        self.sh_irreps = e3nn.Irreps.spherical_harmonics(max_ell)[1:]  # discard 0e

        if interaction_irreps == "o3_restricted":
            self.interaction_irreps = e3nn.Irreps.spherical_harmonics(max_ell)
        elif interaction_irreps == "o3_full":
            self.interaction_irreps = e3nn.Irreps(e3nn.Irrep.iterator(max_ell))
        else:
            self.interaction_irreps = e3nn.Irreps(interaction_irreps)

        self.r_max = r_max
        self.correlation = correlation
        self.avg_num_neighbors = avg_num_neighbors
        self.epsilon = epsilon
        self.readout_mlp_irreps = readout_mlp_irreps
        self.activation = gate
        self.num_interactions = num_interactions
        self.output_irreps = output_irreps
        self.num_species = num_species
        self.symmetric_tensor_product_basis = symmetric_tensor_product_basis
        self.off_diagonal = off_diagonal

        # Embeddings
        self.node_embedding = node_embedding(
            self.num_species, self.num_features * self.hidden_irreps
        )
        self.radial_embedding = RadialEmbeddingBlock(
            r_max=r_max,
            avg_r_min=avg_r_min,
            basis_functions=radial_basis,
            envelope_function=radial_envelope,
        )

    def __call__(
        self,
        vectors: jnp.ndarray,  # [n_edges, 3]
        node_specie: jnp.ndarray,  # [n_nodes] int between 0 and num_species-1
        senders: jnp.ndarray,  # [n_edges]
        receivers: jnp.ndarray,  # [n_edges]
    ) -> e3nn.IrrepsArray:
        assert vectors.ndim == 2 and vectors.shape[1] == 3
        assert node_specie.ndim == 1
        assert senders.ndim == 1 and receivers.ndim == 1
        assert vectors.shape[0] == senders.shape[0] == receivers.shape[0]

        # Embeddings
        node_feats = self.node_embedding(node_specie).astype(
            vectors.dtype
        )  # [n_nodes, feature * irreps]
        node_feats = profile("embedding: node_feats", node_feats)

        lengths = safe_norm(vectors, axis=-1)

        edge_attrs = e3nn.concatenate(
            [
                self.radial_embedding(lengths),
                e3nn.spherical_harmonics(
                    self.sh_irreps,
                    vectors / lengths[..., None],
                    normalize=False,
                    normalization="component",
                ),
            ]
        )  # [n_edges, irreps]

        edge_attrs = profile("embedding: edge_attrs", edge_attrs)

        # Interactions
        outputs = []
        for i in range(self.num_interactions):
            first = i == 0
            last = i == self.num_interactions - 1

            hidden_irreps = (
                self.hidden_irreps
                if not last
                else self.hidden_irreps.filter(self.output_irreps)
            )

            node_outputs, node_feats = MACELayer(
                first=first,
                last=last,
                num_features=self.num_features,
                interaction_irreps=self.interaction_irreps,
                hidden_irreps=hidden_irreps,
                avg_num_neighbors=self.avg_num_neighbors,
                activation=self.activation,
                num_species=self.num_species,
                epsilon=self.epsilon,
                correlation=self.correlation,
                output_irreps=self.output_irreps,
                readout_mlp_irreps=self.readout_mlp_irreps,
                symmetric_tensor_product_basis=self.symmetric_tensor_product_basis,
                off_diagonal=self.off_diagonal,
                name=f"layer_{i}",
            )(
                node_feats,
                node_specie,
                edge_attrs,
                senders,
                receivers,
            )

        return node_outputs  # [n_nodes, output_irreps]