import chex
import jax
import jax.numpy as jnp
import haiku as hk


def equivariant_fn(x, mlp_units=(5,), zero_init: bool = True):
    chex.assert_rank(x, 2)
    diff_combos = x - x[:, None]  # [n_nodes, n_nodes, dim]
    norms = jnp.linalg.norm(diff_combos, ord=2, axis=-1)
    net = hk.Sequential([hk.nets.MLP((mlp_units), activation=jax.nn.elu),
                         hk.Linear(1, w_init=jnp.zeros, b_init=jnp.zeros) if zero_init else
                         hk.Linear(1)])
    m = jnp.squeeze(net(norms[..., None]), axis=-1)
    return x + jnp.einsum('ijd,ij->id', diff_combos / (norms + 1)[..., None], m)

def invariant_fn(x, n_vals, zero_init: bool = True):
    chex.assert_rank(x, 2)
    equivariant_x = jnp.stack([equivariant_fn(x, zero_init=zero_init) for _ in range(n_vals)], axis=-1)
    return jnp.linalg.norm(x[..., None] - equivariant_x, ord=2, axis=-2)
