from typing import NamedTuple, Optional

import haiku as hk
import jax.numpy as jnp
import e3nn_jax as e3nn
import chex

from nets.flex_mace import FlexMACE
from utils.graph import get_senders_and_receivers_fully_connected

class MACETorsoConfig(NamedTuple):
    n_vectors_residual: int
    n_invariant_feat_residual: int
    n_vectors_hidden_readout_block: int
    n_invariant_hidden_readout_block: int
    hidden_irreps: str
    num_features: int = 1
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
    if config.torso_config.num_features != 1:
        raise Exception("This effect is currently unclear to the user.")
    kwargs.update(
        output_irreps=e3nn.Irreps(f"{config.n_invariant_feat_out}x0e+{config.n_vectors_out}x1o"),
        mace_layer_output_irreps=e3nn.Irreps(f"{config.torso_config.n_invariant_feat_residual}x0e+"
                                             f"{config.torso_config.n_vectors_residual}x1o"),
        hidden_irreps=e3nn.Irreps(config.torso_config.hidden_irreps),
        readout_mlp_irreps=e3nn.Irreps(
            f"{config.torso_config.n_invariant_hidden_readout_block}x0e+"
            f"{config.torso_config.n_vectors_hidden_readout_block}x1o"),
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



    def __call__(self, x: chex.Array, h: Optional[chex.Array] = None):
        node_specie = h
        if h is None:
            # No node feature, so initialise them zeros.
            node_specie = jnp.zeros(x.shape[:-1], dtype=int)
        if len(x.shape) == 2:
            return self.call_single(x, node_specie)
        else:
            return hk.vmap(self.call_single, split_rng=False)(x, h, node_specie)

    def call_single(self, x, node_specie):
        """We manually keep track of the centre of mass to ensure translation equivariance."""
        chex.assert_rank(x, 2)

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