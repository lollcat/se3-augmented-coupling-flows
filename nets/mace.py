from typing import NamedTuple, Optional

import haiku as hk
import jax.numpy as jnp
import e3nn_jax as e3nn
import chex

from nets.flex_mace import FlexMACE
from utils.graph import get_senders_and_receivers_fully_connected

class MACETorsoConfig(NamedTuple):
    n_vectors_mace_lay_out: int
    n_invariant_feat_mace_lay_out: int
    n_vectors_readout: int
    n_invariant_feat_readout: int
    n_vectors_hidden: int
    n_invariant_feat_hidden: int
    num_features: int
    num_species: int = 1
    # Maximum angular momentum in the spherical expansion on edges, :math:`l = 0, 1, \dots`.
    # Controls the resolution of the spherical expansion.
    max_ell: int = 5  # Max spherical harmonic degree, default 5 for generative modelling
    num_layers: int = 2  # Number of interactions (layers), default 2
    correlation: int = 3  # Correlation order at each layer (~ node_features^correlation), default 3
    # Average number of neighbours. Used for normalization. Defaults to n-nodes (fully connected).
    avg_num_neighbors: Optional[int] = None
    interaction_mlp_depth: int = 3
    interaction_mlp_width: int = 64

    # Residual MLP  params
    residual_mlp_width: int = 64
    residual_mlp_depth: int = 1

class MACEConfig(NamedTuple):
    name: str
    n_invariant_feat_out: int
    n_vectors_out: int
    zero_init_invariant_feat: bool
    torso_config: MACETorsoConfig


def get_mace_kwargs(config: MACEConfig, avg_num_neighbours: int):
    kwargs = {}
    if config.torso_config.avg_num_neighbors is not None:
        raise Exception("Haven't thought about this yet")
    kwargs.update(
        output_irreps=e3nn.Irreps(f"{config.n_invariant_feat_out}x0e+{config.n_vectors_out}x1o"),
        mace_layer_output_irreps=e3nn.Irreps(f"{config.torso_config.n_invariant_feat_mace_lay_out}x0e+"
                                             f"{config.torso_config.n_vectors_mace_lay_out}x1o"),
        hidden_irreps=e3nn.Irreps(
            f"{config.torso_config.n_invariant_feat_hidden}x0e+{config.torso_config.n_vectors_hidden}x1o"),
        readout_mlp_irreps=e3nn.Irreps(
            f"{config.torso_config.n_invariant_feat_readout}x0e+{config.torso_config.n_vectors_readout}x1o"),
        num_features = config.torso_config.num_features,
        avg_num_neighbors=avg_num_neighbours,
        max_ell = config.torso_config.max_ell,
        num_layers = config.torso_config.num_layers,
        correlation = config.torso_config.correlation,
        interaction_mlp_width = config.torso_config.interaction_mlp_width,
        residual_mlp_width = config.torso_config.residual_mlp_width,
        residual_mlp_depth= config.torso_config.residual_mlp_depth,
    )
    return kwargs



class MaceNet(hk.Module):
    """A wrapper for MACE."""
    def __init__(self, config: MACEConfig):
        super().__init__(name=config.name)
        self.config = config



    def __call__(self, x):
        if len(x.shape) == 2:
            return self.call_single(x)
        else:
            return hk.vmap(self.call_single, split_rng=False)(x)

    def call_single(self, x):
        """We manually keep track of the centre of mass to ensure translation equivariance."""
        chex.assert_rank(x, 2)
        # vectors: jnp.ndarray,  # [n_edges, 3]
        # node_specie: jnp.ndarray,  # [n_nodes] int between 0 and num_species-1
        node_specie = jnp.zeros(x.shape[0], dtype=int)

        # avg_num_neighbors defaults to fully connected.
        avg_num_neighbors = self.config.torso_config.avg_num_neighbors if self.config.torso_config.avg_num_neighbors \
            else x.shape[0]
        mace_fn = FlexMACE(**get_mace_kwargs(self.config, avg_num_neighbors))
        senders, receivers = get_senders_and_receivers_fully_connected(x.shape[0])
        mace_output_irreps = mace_fn(
            positions=x, node_specie=node_specie, senders=senders, receivers=receivers,
            shared_features=jnp.zeros(1)
        )
        vector_features = mace_output_irreps.filter(keep=f"{self.config.n_vectors_out}x1o")
        vector_features = vector_features.factor_mul_to_last_axis()  # [n_nodes, n_vectors, dim]
        vector_features = vector_features.array
        invariant_features = mace_output_irreps.filter(keep=f"{self.config.n_invariant_feat_out}x0e")

        invariant_features = hk.Linear(invariant_features.shape[-1],
                                       w_init=jnp.zeros if self.config.zero_init_invariant_feat else None,
                                       )(invariant_features.array)
        return vector_features, invariant_features