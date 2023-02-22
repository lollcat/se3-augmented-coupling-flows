import jax.numpy as jnp

def get_senders_and_receivers_fully_connected(n_nodes: int):
    senders = []
    receivers = []
    for i in range(n_nodes):
        for j in range(n_nodes - 1):
            senders.append(i)
            receivers.append((i + 1 + j) % n_nodes)
    return jnp.array(senders), jnp.array(receivers)