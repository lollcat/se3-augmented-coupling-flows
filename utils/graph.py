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


def get_x_flattened_over_multiplicity_and_senders_recievers(x: chex.Array) -> Tuple[chex.Array, chex.Array, chex.Array]:
    n_nodes, multiplicity, dim = x.shape
    senders, receivers = get_senders_and_receivers_fully_connected(n_nodes)
    x = jnp.reshape(x, (n_nodes * multiplicity, dim))
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
    return x, senders, receivers
