from typing import Callable

import jax.numpy as jnp
import e3nn_jax as e3nn
def get_senders_and_receivers_fully_connected(n_nodes: int):
    senders = []
    receivers = []
    for i in range(n_nodes):
        for j in range(n_nodes - 1):
            senders.append(i)
            receivers.append((i + 1 + j) % n_nodes)
    return jnp.array(senders), jnp.array(receivers)

def e3nn_apply_activation(x: e3nn.IrrepsArray, activation_fn: Callable):
    assert x.irreps.is_scalar()
    return e3nn.IrrepsArray(irreps=x.irreps, array=activation_fn(x.array))
