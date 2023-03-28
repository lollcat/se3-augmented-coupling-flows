from typing import NamedTuple, Callable, Sequence

import haiku as hk
import jax.numpy as jnp
import jax
import e3nn_jax as e3nn
from mace_jax.modules.models import safe_norm
import chex
import numpy as np

from molboil.models.base import EquivariantForwardFunction

from utils.graph import get_senders_and_receivers_fully_connected

class EGCL(hk.Module):
    """A version of EGCL coded only with haiku (not e3nn) so works for arbitary dimension of inputs."""

    def __init__(
        self,
        name: str,
        mlp_units: Sequence[int],
        n_invariant_feat_hidden: int,
        activation_fn: Callable,
        residual_h: bool,
        residual_x: bool,
        normalization_constant: float,
        variance_scaling_init: float,
        cross_multiplicty_node_feat: bool,
    ):
        """_summary_

        Args:
            name (str)
            mlp_units (Sequence[int]): sizes of hidden layers for all MLPs
            residual_h (bool): whether to use a residual connectio probability density for scalars
            residual_x (bool): whether to use a residual connectio probability density for vectors.
            get_shifts_via_tensor_product (bool): Whether to use tensor product for message construction
            variance_scaling_init (float): Value to scale the output variance of MLP multiplying message vectors
            cross_multiplicty_node_feat (bool): Whether to use cross multiplicity for node features.
        """
        super().__init__(name=name)
        self.variance_scaling_init = variance_scaling_init
        self.mlp_units = mlp_units
        self.n_invariant_feat_hidden = n_invariant_feat_hidden
        self.activation_fn = activation_fn
        self.residual_h = residual_h
        self.residual_x = residual_x
        self.normalization_constant = normalization_constant
        self.cross_multiplicty_node_feat = cross_multiplicty_node_feat


        self.phi_e = hk.nets.MLP(mlp_units, activation=activation_fn, activate_final=True)
        self.phi_inf = lambda x: jax.nn.sigmoid(hk.Linear(1)(x))

        self.phi_x_torso = hk.nets.MLP(mlp_units, activate_final=True, activation=activation_fn)
        self.phi_h = hk.nets.MLP((*mlp_units, self.n_invariant_feat_hidden), activate_final=False,
                                 activation=activation_fn)

    def __call__(self, node_positions, node_features, senders, receivers):
        """E(N)GNN layer implemented with E3NN package

        Args:
            node_positions [n_nodes, self.n_vectors_hidden, 3]-ndarray: augmented set of euclidean coodinates for each node
            node_features [n_nodes, self.n_invariant_feat_hidden]-ndarray: scalar features at each node

        Returns:
            vectors_out [n_nodes, self.n_vectors_hidden, 3]-ndarray: augmented set of euclidean coodinates for each node
            features_out [n_nodes, self.n_invariant_feat_hidden]-ndarray: scalar features at each node
            senders:
            receivers:
        """
        n_nodes, n_vectors, dim = node_positions.shape
        avg_num_neighbours = n_nodes - 1
        chex.assert_tree_shape_suffix(node_features, (self.n_invariant_feat_hidden,))

        # Prepare the edge attributes.
        vectors = node_positions[receivers] - node_positions[senders]
        lengths = safe_norm(vectors, axis=-1, keepdims=False)
        sq_lengths = lengths ** 2

        edge_feat_in = jnp.concatenate([node_features[senders], node_features[receivers], sq_lengths], axis=-1)

        if self.cross_multiplicty_node_feat:
            senders_cross, recievers_cross = get_senders_and_receivers_fully_connected(n_vectors)
            cross_vectors = node_positions[:, recievers_cross] - node_positions[:, senders_cross]
            chex.assert_shape(cross_vectors, (n_nodes, n_vectors, dim))
            cross_sq_lengths = jnp.sum(cross_vectors ** 2, axis=-1)
            edge_feat_in = jnp.concatenate([edge_feat_in, cross_sq_lengths[senders], cross_sq_lengths[receivers]],
                                           axis=-1)

        # build messages
        m_ij = self.phi_e(edge_feat_in)

        # Get positional output
        phi_x_out = self.phi_x_torso(m_ij)
        phi_x_out = hk.Linear(
            n_vectors, w_init=hk.initializers.VarianceScaling(self.variance_scaling_init, "fan_avg", "uniform")
        )(phi_x_out)

        shifts_ij = (
            phi_x_out[:, :, None]
            * vectors
            / (self.normalization_constant + lengths[:, :, None])
        )  # scale vectors by messages and
        shifts_i = e3nn.scatter_sum(
            data=shifts_ij, dst=receivers, output_size=n_nodes
        )
        vectors_out = shifts_i / avg_num_neighbours
        chex.assert_equal_shape((vectors_out, node_positions))

        # Get feature output
        e = self.phi_inf(m_ij)
        m_i = e3nn.scatter_sum(
            data=m_ij*e, dst=receivers, output_size=n_nodes
        ) / jnp.sqrt(avg_num_neighbours)
        phi_h_in = jnp.concatenate([m_i, node_features], axis=-1)
        features_out = self.phi_h(phi_h_in)
        chex.assert_equal_shape((features_out, node_features))

        # Final processing and conversion into plain arrays.

        if self.residual_h:
            features_out = features_out + node_features
        if self.residual_x:
            vectors_out = node_positions + vectors_out
        return vectors_out, features_out


class EGNNTorsoConfig(NamedTuple):
    name: str
    n_blocks: int  # number of layers
    mlp_units: Sequence[int]
    n_invariant_feat_hidden: int
    n_vectors_hidden: int = 1  # Typically gets manually overwritten by the flow.
    activation_fn: Callable = jax.nn.silu
    residual_h: bool = True
    residual_x: bool = True
    normalization_constant: float = 1.0
    variance_scaling_init: float = 0.001
    cross_multiplicty_node_feat: bool = True

    def get_EGCL_kwargs(self):
        kwargs = self._asdict()
        del kwargs["n_blocks"]
        del kwargs["name"]
        del kwargs["n_vectors_hidden"]
        return kwargs


def make_egnn_torso_forward_fn(
    torso_config: EGNNTorsoConfig,
) -> EquivariantForwardFunction:
    def forward_fn(
        positions: chex.Array,
        node_features: chex.Array,
        senders: chex.Array,
        receivers: chex.Array,
    ):
        chex.assert_rank(positions, 2)
        chex.assert_rank(node_features, 2)
        chex.assert_rank(senders, 1)
        chex.assert_rank(receivers, 1)

        n_nodes, dim = positions.shape

        vectors = positions - positions.mean(axis=0, keepdims=True)
        vectors = vectors[:, None]
        initial_vectors = vectors

        # Create n-multiplicity copies of h and vectors.
        vectors = jnp.repeat(vectors, torso_config.n_vectors_hidden, axis=1)
        h = hk.Linear(torso_config.n_invariant_feat_hidden)(node_features)

        for i in range(torso_config.n_blocks):
            vectors, h = EGCL(
                torso_config.name + str(i), **torso_config.get_EGCL_kwargs()
            )(vectors, h, senders, receivers)

        if torso_config.residual_x:
            vectors = vectors - initial_vectors
        return vectors, h

    return forward_fn
