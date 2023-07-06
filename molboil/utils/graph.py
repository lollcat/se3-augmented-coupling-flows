from typing import Tuple

import chex
import jax.numpy as jnp

def get_senders_and_receivers_fully_connected(n_nodes: int) -> Tuple[chex.Array, chex.Array]:
    """Get senders and receivers for fully connected graph of `n_nodes`."""
    receivers = []
    senders = []
    for i in range(n_nodes):
        for j in range(n_nodes - 1):
            receivers.append(i)
            senders.append((i + 1 + j) % n_nodes)
    return jnp.array(senders, dtype=int), jnp.array(receivers, dtype=int)

def fill_diagonal(a, val):
    assert a.ndim >= 2
    i, j = jnp.diag_indices(min(a.shape[-2:]))
    return a.at[..., i, j].set(val)