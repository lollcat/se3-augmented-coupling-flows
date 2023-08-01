from typing import Tuple, Optional

import jax.numpy as jnp
import chex


def get_senders_and_receivers_fully_connected(n_nodes: int) -> Tuple[chex.Array, chex.Array]:
    receivers = []
    senders = []
    for i in range(n_nodes):
        for j in range(n_nodes - 1):
            receivers.append(i)
            senders.append((i + 1 + j) % n_nodes)
    return jnp.array(senders), jnp.array(receivers)


def unflatten_vectors_scalars(vectors: chex.Array, scalars: chex.Array,
                              n_nodes: int, multiplicity: int, dim: int) -> Tuple[chex.Array, chex.Array]:
    chex.assert_rank(vectors, 3)
    chex.assert_rank(scalars, 2)
    n_nodes_multp, n_vectors, dim_ = vectors.shape
    n_nodes_multp_, n_scalars = scalars.shape
    assert n_nodes_multp == (n_nodes*multiplicity) == n_nodes_multp_
    assert dim == dim_

    vectors_out = jnp.reshape(vectors, (n_nodes, multiplicity, n_vectors, dim))
    scalars_out = jnp.reshape(scalars, (n_nodes, multiplicity, n_scalars))
    return vectors_out, scalars_out



def get_pos_feat_send_receive_flattened_over_multiplicity(
        positions: chex.Array,
        features: chex.Array,
        senders: Optional[chex.Array] = None,
        receivers: Optional[chex.Array] = None,
        ) -> Tuple[chex.Array, chex.Array, chex.Array, chex.Array]:
    """Flattens features and positions over multiplicity. Including a sender/reciever connection across
    multiplicities (i.e. first node of first multiplicity connected to first node of second multiplicity in graph)."""
    n_nodes, multiplicity, dim = positions.shape

    if senders is None or receivers is None:
        senders, receivers = get_senders_and_receivers_fully_connected(n_nodes)
    chex.assert_tree_shape_prefix(features, (n_nodes, multiplicity))
    n_features = features.shape[-1]

    # Get flat positions and features.
    positions = jnp.reshape(positions, (n_nodes * multiplicity, dim))
    features = jnp.reshape(features, (n_nodes * multiplicity, n_features))

    # Get senders and receivers, with connection across multiplicities.
    senders_list = []
    receivers_list = []
    for i in range(multiplicity):
        # Pairwise within each multiplicity group.
        senders_list.append(senders + i * n_nodes)
        receivers_list.append(receivers + i * n_nodes)

        # Pairwise across multiplicity groups.
        for j in range(multiplicity - 1):
            receivers_list.append(jnp.arange(n_nodes) + i * n_nodes)
            senders_list.append(jnp.arange(n_nodes) + ((j + i + 1) % multiplicity) * n_nodes)
    senders = jnp.concatenate(senders_list)
    receivers = jnp.concatenate(receivers_list)
    return positions, features, senders, receivers


def fill_diagonal(a, val):
    assert a.ndim >= 2
    i, j = jnp.diag_indices(min(a.shape[-2:]))
    return a.at[..., i, j].set(val)
