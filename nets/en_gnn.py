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
    """A version of EGCL coded only with haiku (not e3nn) so works for arbitary dimension of inputs.
    Follows notation of https://arxiv.org/abs/2105.09016. """

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
        cross_multiplicity_shifts: bool,
    ):
        """_summary_

        Args:
            name (str)
            mlp_units (Sequence[int]): sizes of hidden layers for all MLPs
            residual_h (bool): whether to use a residual connectio probability density for scalars
            residual_x (bool): whether to use a residual connectio probability density for vectors.
            normalization_constant (float): Value to normalize the output of MLP multiplying message vectors.
                C in the en normalizing flows paper (https://arxiv.org/abs/2105.09016).
            variance_scaling_init (float): Value to scale the output variance of MLP multiplying message vectors
            cross_multiplicty_node_feat (bool): Whether to use cross multiplicity for node features.
            cross_multiplicity_shifts (bool): Whether to use cross multiplicity for shifts.
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
        self.cross_multiplicity_shifts = cross_multiplicity_shifts


        self.phi_e = hk.nets.MLP(mlp_units, activation=activation_fn, activate_final=True)
        self.phi_inf = lambda x: jax.nn.sigmoid(hk.Linear(1)(x))

        self.phi_x_torso = hk.nets.MLP(mlp_units, activate_final=True, activation=activation_fn)
        self.phi_h = hk.nets.MLP((*mlp_units, self.n_invariant_feat_hidden), activate_final=False,
                                 activation=activation_fn)
        if self.cross_multiplicity_shifts:
            self.phi_x_cross_torso = hk.nets.MLP(mlp_units, activate_final=True, activation=activation_fn)

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
        chex.assert_rank(node_positions, 3)
        chex.assert_rank(node_features, 2)
        chex.assert_rank(senders, 1)
        chex.assert_equal_shape([senders, receivers])
        n_nodes, n_vectors, dim = node_positions.shape
        avg_num_neighbours = n_nodes - 1
        chex.assert_tree_shape_suffix(node_features, (self.n_invariant_feat_hidden,))

        # Prepare the edge attributes.
        vectors = node_positions[receivers] - node_positions[senders]
        lengths = safe_norm(vectors, axis=-1, keepdims=False)
        sq_lengths = lengths ** 2

        edge_feat_in = jnp.concatenate([node_features[senders], node_features[receivers], sq_lengths], axis=-1)

        if (self.cross_multiplicty_node_feat or self.cross_multiplicity_shifts) and n_vectors > 1:
            senders_cross, recievers_cross = get_senders_and_receivers_fully_connected(n_vectors)
            cross_vectors = node_positions[:, recievers_cross] - node_positions[:, senders_cross]
            n_cross_vectors = n_vectors * (n_vectors - 1)
            chex.assert_shape(cross_vectors, (n_nodes, n_cross_vectors, dim))
            if self.cross_multiplicty_node_feat:
                cross_lengths = safe_norm(cross_vectors, axis=-1, keepdims=False)
                cross_sq_lengths = cross_lengths**2  # node features [n_nodes, n_cross_vectors]
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

        if self.cross_multiplicity_shifts and n_vectors > 1:
            phi_cross_in = e3nn.scatter_sum(m_ij, dst=senders, output_size=n_nodes)  # [n_nodes, feat]
            phi_x_cross_out = self.phi_x_cross_torso(phi_cross_in)
            phi_x_cross_out = hk.Linear(
                n_cross_vectors, w_init=hk.initializers.VarianceScaling(self.variance_scaling_init, "fan_avg", "uniform")
            )(phi_x_cross_out)
            cross_shifts_im = (
                phi_x_cross_out[:, :, None]
                * cross_vectors
                / (self.normalization_constant + cross_sq_lengths[:, :, None])
            )
            cross_shifts_i = e3nn.scatter_sum(jnp.swapaxes(cross_shifts_im, 0, 1), dst=recievers_cross, output_size=n_vectors)
            cross_shifts_i = jnp.swapaxes(cross_shifts_i, 0, 1)
            chex.assert_equal_shape((cross_shifts_i, vectors_out))
            vectors_out = vectors_out + cross_shifts_i / (n_vectors - 1)

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
    n_vectors_hidden_per_vec_in: int = 1
    activation_fn: Callable = jax.nn.silu
    residual_h: bool = True
    residual_x: bool = True
    normalization_constant: float = 1.0
    variance_scaling_init: float = 0.001
    cross_multiplicty_node_feat: bool = True
    cross_multiplicity_shifts: bool = True

    def get_EGCL_kwargs(self, i):
        kwargs = self._asdict()
        del kwargs["n_blocks"]
        kwargs["name"] = kwargs["name"] + f"_{i}"
        del kwargs["n_vectors_hidden_per_vec_in"]
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
        chex.assert_rank(positions, 3)
        chex.assert_rank(node_features, 2)
        chex.assert_rank(senders, 1)
        chex.assert_rank(receivers, 1)

        n_nodes, vec_multiplicity_in, dim = positions.shape

        # Setup torso input.
        vectors = positions - positions.mean(axis=0, keepdims=True)
        # Create n-multiplicity copies of h and vectors.
        vectors = jnp.repeat(vectors, torso_config.n_vectors_hidden_per_vec_in, axis=1)
        initial_vectors = vectors
        h = hk.Linear(torso_config.n_invariant_feat_hidden)(node_features)

        # Loop through torso layers.
        for i in range(torso_config.n_blocks):
            vectors, h = EGCL(**torso_config.get_EGCL_kwargs(i)
            )(vectors, h, senders, receivers)

        if torso_config.residual_x:
            vectors = vectors - initial_vectors

        chex.assert_shape(vectors, (n_nodes, vec_multiplicity_in*torso_config.n_vectors_hidden_per_vec_in, dim))
        chex.assert_shape(h, (n_nodes, torso_config.n_invariant_feat_hidden))
        return vectors, h

    return forward_fn
