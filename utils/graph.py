from typing import Callable, Tuple

import jax.numpy as jnp
import chex
import e3nn_jax as e3nn

def get_senders_and_receivers_fully_connected(n_nodes: int):
    receivers = []
    senders = []
    for i in range(n_nodes):
        for j in range(n_nodes - 1):
            receivers.append(i)
            senders.append((i + 1 + j) % n_nodes)
    return jnp.array(senders), jnp.array(receivers)

def e3nn_apply_activation(x: e3nn.IrrepsArray, activation_fn: Callable):
    assert x.irreps.is_scalar()
    return e3nn.IrrepsArray(irreps=x.irreps, array=activation_fn(x.array))


def get_pos_feat_send_receive_flattened_over_multiplicity(positions: chex.Array,
                                                          features: chex.Array
                                                          ) -> Tuple[chex.Array, chex.Array, chex.Array, chex.Array]:
    n_nodes, multiplicity, dim = positions.shape
    chex.assert_tree_shape_prefix(features, (n_nodes, multiplicity))
    n_features = features.shape[-1]

    # Get flat positions and features.
    positions = jnp.reshape(positions, (n_nodes * multiplicity, dim))
    features = jnp.reshape(features, (n_nodes * multiplicity, n_features))

    # Get senders and receivers, with connection across multiplicities.
    senders, receivers = get_senders_and_receivers_fully_connected(n_nodes)
    senders_list = []
    receivers_list = []
    for i in range(multiplicity):
        # Pairwise within each multiplicity group.
        senders_list.append(senders + i * dim)
        receivers_list.append(receivers + i * dim)

        # Pairwise across multiplicity groups.
        for j in range(multiplicity - 1):
            receivers_list.append(jnp.arange(dim) + i * dim)
            senders_list.append(jnp.arange(dim) + ((j + i + 1) % multiplicity) * dim)
    senders = jnp.concatenate(senders_list)
    receivers = jnp.concatenate(receivers_list)
    return positions, features, senders, receivers
