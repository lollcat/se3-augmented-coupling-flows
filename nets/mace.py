from typing import NamedTuple

import haiku as hk
import jax.numpy as jnp
from mace_jax.modules.models import MACE
import e3nn_jax as e3nn

class MACEConfig(NamedTuple):
    name: str
    n_invariant_feat_readout: int
    n_vectors_readout: int
    n_invariant_feat_lay_out: int
    n_vectors_lay_out: int
    n_vectors_hidden: int
    n_invariant_feat_hidden: int
    avg_num_neighbors: int
    num_species: int = 1


def get_vectors(x):
    vectors = x - x[None, ...]
    vectors = vectors[jnp.tril(vectors.shape[0])]
    return vectors


class MaceNet(hk.Module):
    """A wrapper for MACE."""
    def __init__(self, config: MACEConfig):
        super().__init__(name=config.name)
        self.mace = MACE(
            output_irreps=e3nn.Irreps(f"{config.n_invariant_feat_lay_out}x0e+{config.n_vectors_lay_out}x1e"),
            r_max=10.0,
            num_interactions=2,
            hidden_irreps=e3nn.Irreps(f"{config.n_invariant_feat_hidden}x0e+{config.n_vectors_hidden}x1e"),
            readout_mlp_irreps=e3nn.Irreps(f"{config.n_invariant_feat_readout}x0e+{config.n_vectors_readout}x1e"),
            avg_num_neighbors=config.avg_num_neighbors,
            num_species=config.num_species,
            radial_basis=jnp.array,
            radial_envelope=jnp.array

        )

    def __call__(self, x):
        # vectors: jnp.ndarray,  # [n_edges, 3]
        # node_specie: jnp.ndarray,  # [n_nodes] int between 0 and num_species-1
        # senders: jnp.ndarray,  # [n_edges]
        # receivers: jnp.ndarray,  # [n_edges]
        vectors = get_vectors(x)
        node_specie = jnp.zeros(x.shape[0])
