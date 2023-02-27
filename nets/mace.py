from typing import NamedTuple, Optional, Union
from functools import partial

import haiku as hk
import jax.numpy as jnp
import jax
from mace_jax.data import get_neighborhood
from mace_jax import tools
import e3nn_jax as e3nn
import chex

from nets.mace_net_adjusted import MACE
from utils.graph import get_senders_and_receivers_fully_connected

class MACETorsoConfig(NamedTuple):
    n_vectors_hidden: int
    n_invariant_feat_hidden: int
    bessel_number: int  # Number of bessel functions.
    r_max: float  # Used in bessel function.
    num_species: int = 1
    # Maximum angular momentum in the spherical expansion on edges, :math:`l = 0, 1, \dots`.
    # Controls the resolution of the spherical expansion.
    n_interactions: int = 2
    cut_off: float = 1.e6  # No cutoff by default for fully connected graph. Currently unused.
    # Average number of neighbours. Used for normalization. Defaults to n-nodes (fully connected).
    avg_num_neighbors: Optional[int] = None
    interaction_irreps: Union[str, e3nn.Irreps] = "o3_restricted" # or o3_full


class MACEConfig(NamedTuple):
    name: str
    n_invariant_feat_readout: int
    n_vectors_readout: int
    zero_init_invariant_feat: bool
    torso_config: MACETorsoConfig


class MaceNet(hk.Module):
    """A wrapper for MACE."""
    def __init__(self, config: MACEConfig):
        super().__init__(name=config.name)
        self.config = config
        # TODO: setting lay out irreps to match read out. That makes sense?
        # TODO: A sprinkle of MLPs



    def __call__(self, x):
        if len(x.shape) == 2:
            return self.call_single(x)
        else:
            return hk.vmap(self.call_single, split_rng=False)(x)

    def call_single(self, x):
        """We manually keep track of the centre of mass to ensure translation equivariance."""
        centre_of_mass = jnp.mean(x, axis=-2)
        chex.assert_rank(x, 2)
        # vectors: jnp.ndarray,  # [n_edges, 3]
        # node_specie: jnp.ndarray,  # [n_nodes] int between 0 and num_species-1
        node_specie = jnp.zeros(x.shape[0], dtype=int)

        # avg_num_neighbors defaults to fully connected.
        avg_num_neighbors = self.config.torso_config.avg_num_neighbors if self.config.torso_config.avg_num_neighbors \
            else x.shape[0]
        mace_fn = MACE(
            output_irreps=e3nn.Irreps(f"{self.config.n_invariant_feat_readout}x0e+{self.config.n_vectors_readout}x1o"),
            r_max=self.config.torso_config.r_max,
            num_interactions=self.config.torso_config.n_interactions,
            hidden_irreps=e3nn.Irreps(f"{self.config.torso_config.n_invariant_feat_hidden}x0e+{self.config.torso_config.n_vectors_hidden}x1o"),
            readout_mlp_irreps=e3nn.Irreps(f"{self.config.n_invariant_feat_readout}x0e+{self.config.n_vectors_readout}x1o"),
            avg_num_neighbors=avg_num_neighbors,
            num_species=self.config.torso_config.num_species,
            radial_basis=lambda length, max_length: e3nn.bessel(length, x_max=max_length,
                                                                n=self.config.torso_config.bessel_number),
            radial_envelope=e3nn.soft_envelope,
            interaction_irreps=self.config.torso_config.interaction_irreps

        )
        # senders, receivers, shifts = get_neighborhood(
        #     positions=x, cutoff=self.config.layer_config.cut_off, pbc=None, cell=None
        # )
        # TODO: Above doesn't jit, so do manually below. Also need to understand what shifts is, and the arguments
        # into the above function.
        senders, receivers = get_senders_and_receivers_fully_connected(x.shape[0])
        # As cell is set to None, shifts is not used in
        shifts = jnp.zeros_like(x) * jnp.nan

        vectors = tools.get_edge_relative_vectors(
            positions=x,
            senders=senders,
            receivers=receivers,
            shifts=shifts,
            cell=None,
            n_edge=senders.shape[0],
        )

        mace_output_irreps = mace_fn(
            vectors=vectors, node_specie=node_specie, senders=senders, receivers=receivers)
        vector_features = mace_output_irreps.filter(keep=f"{self.config.n_vectors_readout}x1o")
        vector_features = vector_features.factor_mul_to_last_axis()  # [n_nodes, n_vectors, dim]
        vector_features = vector_features.array
        invariant_features = mace_output_irreps.filter(keep=f"{self.config.n_invariant_feat_readout}x0e")

        invariant_features = hk.Linear(invariant_features.shape[-1],
                                       w_init=jnp.zeros if self.config.zero_init_invariant_feat else None,
                                       )(jax.nn.elu(invariant_features.array))

        return vector_features + centre_of_mass[:, None], invariant_features