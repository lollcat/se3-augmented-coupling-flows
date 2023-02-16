from typing import NamedTuple
from functools import partial

import haiku as hk
import jax.numpy as jnp
from nets.mace_net_adjusted import MACE
from mace_jax.tools.gin_model import bessel_basis, soft_envelope
from mace_jax.data import get_neighborhood
from mace_jax import tools
import e3nn_jax as e3nn
import chex

class MACEConfig(NamedTuple):
    name: str
    n_invariant_feat_readout: int
    n_vectors_readout: int
    n_vectors_hidden: int
    n_invariant_feat_hidden: int
    avg_num_neighbors: int
    r_max: float
    num_species: int = 1
    n_interactions: int = 2
    cut_off: float = 1.e6  # No cutoff by default for graph connection.
    bessel_number: int = 8


class MaceNet(hk.Module):
    """A wrapper for MACE."""
    def __init__(self, config: MACEConfig):
        super().__init__(name=config.name)
        self.config = config
        # TODO: setting lay out irreps to match read out. That makes sense?
        self.mace_fn = MACE(
            output_irreps=e3nn.Irreps(f"{config.n_invariant_feat_readout}x0e+{config.n_vectors_readout}x1e"),
            r_max=config.r_max,
            num_interactions=config.n_interactions,
            hidden_irreps=e3nn.Irreps(f"{config.n_invariant_feat_hidden}x0e+{config.n_vectors_hidden}x1e"),
            readout_mlp_irreps=e3nn.Irreps(f"{config.n_invariant_feat_readout}x0e+{config.n_vectors_readout}x1e"),
            avg_num_neighbors=config.avg_num_neighbors,
            num_species=config.num_species,
            radial_basis=partial(bessel_basis, number=config.bessel_number),
            radial_envelope=soft_envelope
        )

    def __call__(self, x):
        """We manually keep track of the centre of mass to ensure translation equivariance."""
        centre_of_mass = jnp.mean(x, axis=-2)
        chex.assert_rank(x, 2)
        # vectors: jnp.ndarray,  # [n_edges, 3]
        # node_specie: jnp.ndarray,  # [n_nodes] int between 0 and num_species-1
        node_specie = jnp.zeros(x.shape[0], dtype=int)
        senders, receivers, shifts = get_neighborhood(
            positions=x, cutoff=self.config.cut_off, pbc=None, cell=None
        )  # TODO: understand pbc and cell arguments.
        vectors = tools.get_edge_relative_vectors(
            positions=x,
            senders=senders,
            receivers=receivers,
            shifts=shifts,
            cell=None,
            n_edge=senders.shape[0],
        )

        mace_output_irreps = self.mace_fn(
            vectors=vectors, node_specie=node_specie, senders=senders, receivers=receivers)
        vector_features = mace_output_irreps.filter(keep=f"{self.config.n_vectors_readout}x1e")
        vector_features = vector_features.factor_mul_to_last_axis()  # [n_nodes, n_vectors, dim]
        invariant_features = mace_output_irreps.filter(keep=f"{self.config.n_invariant_feat_readout}x0e")
        return invariant_features.array, vector_features.array + centre_of_mass
