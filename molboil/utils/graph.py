from typing import Tuple

import chex
import jax.numpy as jnp
import jax
import e3nn_jax as e3nn

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


def nearest_neighbors_jax(X, k):
    pdist = jnp.sum((X[:, None, :] - X[None, :, :]) ** 2, axis=-1)
    pdist = fill_diagonal(pdist, jnp.inf * jnp.ones((X.shape[0])))
    return jax.lax.top_k(-pdist, k)[1]



def get_edges_knn(x, k):
    senders = nearest_neighbors_jax(x, k=k).reshape(-1)
    receivers = jnp.arange(0, x.shape[-2], 1)
    receivers = jnp.repeat(receivers, k, 0).reshape(-1)
    return senders, receivers


def get_edge_attr(edge_length, max_radius, n_basis, radial_basis):
    edge_attr = (
        e3nn.soft_one_hot_linspace(
            edge_length,
            start=0.0,
            end=max_radius,
            number=n_basis,
            basis=radial_basis,
            cutoff=False,
        )
        * n_basis**0.5
        * 0.95
    )
    edge_weight_cutoff = 1.4 * e3nn.sus(10 * (1 - edge_length / max_radius))
    # edge_weight_cutoff = e3nn.sus(3.0 * (2.0 - edge_length))
    edge_attr *= edge_weight_cutoff[:, None]
    return edge_attr, edge_weight_cutoff
